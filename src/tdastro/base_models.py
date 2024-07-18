"""The base models used to specify the TDAstro computation graph.

The computation graph is composed of ParameterizedNodes which store variables
and encode the dependencies between individual attributes. We say that variable
X is dependent on variable Y if the value of Y is necessary to compute the value
of X. Thus dependencies impose an ordering of variables in the graph. Y must
be computed before X.

All dynamic attributes (variables whose values change when the graph is resampled)
in the graph must be added using ParameterizedNode.add_parameter(). This allows the graph
to track which variables to update and how to set them. Attributes can be set from a few
sources:
1) A constant
2) A static function or method (which does not have variables that are resampled).
3) The result of evaluating a FunctionNode, which provides a computation using other
   variables in the graph.
4) The attribute of another ParameterizedNode.

The execution graph is processed by starting at the final node, examining each
attribute, and recursively proceeding 'up' the graph for any attribute that
has a dependency. For example the function.

f(a, b) = x
g(c) = y
h(x, y) = z

would form the graph:

a -\
    x - \
b -/     \
          z
c -- y -- /

where z is the 'bottom' node. Attributes a, b, and c would be at the 'top' of the
graph because they have no dependencies.  Such attributes are set by constants or
static functions.
"""

import types
from enum import Enum
from hashlib import md5
from os import urandom


class ParameterSource(Enum):
    """ParameterSource specifies where a ParameterizedNode should get the value
    for a given parameter: a constant value or from another ParameterizedNode.
    """

    CONSTANT = 1
    FUNCTION = 2
    MODEL_ATTRIBUTE = 3
    FUNCTION_NODE = 4


class ParameterizedNode:
    """Any model that uses parameters that can be set by constants,
    functions, or other parameterized nodes.

    Attributes
    ----------
    node_identifier : `str`
        An identifier (or name) for the current node.
    setters : `dict` of `tuple`
        A dictionary to information about the setters for the parameters in the form:
        (ParameterSource, setter information, required). The attributes are
        stored in the order in which they need to be set.
    sample_iteration : `int`
        A counter used to syncronize sampling runs. Tracks how many times this
        model's parameters have been resampled.
    _object_seed : `int`
        A object-specific seed to control random number generation.
    _graph_base_seed, `int`
        A base random seed to use for this specific evaluation graph. Used
        for validity checking.

    Parameters
    ----------
    node_identifier : `str`, optional
        An identifier (or name) for the current node.
    graph_base_seed : `int`, optional
        A base random seed to use for this specific evaluation graph.
        WARNING: This seed should almost never be set manually. Using the same
        seed for multiple graph instances will produce biased samples.
        If set to ``None`` will use urandom() to produce a fully random seed.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, node_identifier=None, graph_base_seed=None, **kwargs):
        self.setters = {}
        self.sample_iteration = 0
        self.node_identifier = node_identifier
        self.set_graph_base_seed(graph_base_seed)

    def __str__(self):
        """Return the string representation of the node."""
        name = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        if self.node_identifier:
            return f"{self.node_identifier}={name}"
        else:
            return name

    def set_graph_base_seed(self, graph_base_seed):
        """Set a new graph base seed.

        Notes
        -----
        WARNING: This seed should almost never be set manually. Using the same
        seed for multiple graph instances will produce biased samples.

        Parameters
        ----------
        graph_base_seed : `int`, optional
            A base random seed to use for this specific evaluation graph.
        """
        if graph_base_seed is None:
            graph_base_seed = int.from_bytes(urandom(4), "big")
        self._graph_base_seed = graph_base_seed

        hashed_object_name = md5(str(self).encode())
        seed_offset = int(hashed_object_name.hexdigest(), base=16)
        self._object_seed = (graph_base_seed + seed_offset) % (2**31)

    def check_resample(self, other):
        """Check if we need to resample the current node based
        on the state of another node trying to access its attributes
        or methods.

        Parameters
        ----------
        other : `ParameterizedNode`
            The node that is accessing the attribute or method
            of the current node.

        Returns
        -------
        bool
            Indicates whether to resample or not.

        Raises
        ------
        ``ValueError`` if the graph has gotten out of sync.
        """
        if other == self:
            return False
        if other.sample_iteration == self.sample_iteration:
            return False
        if other.sample_iteration != self.sample_iteration + 1:
            raise ValueError(
                f"Node {str(other)} at iteration {other.sample_iteration} accessing"
                f" parent {str(self)} at iteration {self.sample_iteration}."
            )
        return True

    def set_parameter(self, name, value=None, **kwargs):
        """Set a single *existing* parameter to the ParameterizedNode.

        Notes
        -----
        * Sets an initial value for the attribute based on the given information.
        * The attributes are stored in the order in which they are added.

        Parameters
        ----------
        name : `str`
            The parameter name to add.
        value : any, optional
            The information to use to set the parameter. Can be a constant,
            function, ParameterizedNode, or self.
        **kwargs : `dict`, optional
           All other keyword arguments, possibly including the parameter setters.

        Raises
        ------
        Raise a ``KeyError`` if there is a parameter collision or the parameter
        cannot be found.
        Raise a ``ValueError`` if the setter type is not supported.
        Raise a ``ValueError`` if the parameter is required, but set to None.
        """
        # Check for parameter has been added and if so, find the index.
        if name not in self.setters:
            raise KeyError(f"Tried to set parameter {name} that has not been added.") from None
        required = self.setters[name][2]

        if value is None and name in kwargs:
            # The value wasn't set, but the name is in kwargs.
            value = kwargs[name]

        if callable(value):
            if isinstance(value, types.FunctionType):
                # Case 1a: This is a static function (not attached to an object).
                self.setters[name] = (ParameterSource.FUNCTION, value, required)
            elif isinstance(value.__self__, ParameterizedNode):
                # Case 1b: This is a method attached to another ParameterizedNode.
                # We do NOT allow these since it prevents tracking an accurate
                # dependency graph.
                raise ValueError(
                    "Cannot set the attribute of a ParameterizedNode from the method "
                    "of another ParameterizedNode. Use FunctionNode instead."
                )
            else:
                # Case 1c: This is a general callable method from another object.
                # We treat it as static (we don't resample the other object).
                self.setters[name] = (ParameterSource.FUNCTION, value, required)
            setattr(self, name, value(**kwargs))
        elif isinstance(value, FunctionNode):
            # Case 2: We are using the result of a computation of the function node.
            self.setters[name] = (ParameterSource.FUNCTION_NODE, value, required)
            setattr(self, name, value.compute(**kwargs))
        elif isinstance(value, ParameterizedNode):
            # Case 3a: We are trying to access a parameter of a ParameterizedNode
            # with the same name.
            if not hasattr(value, name):
                raise ValueError(f"Attribute {name} missing from {str(value)}.")
            elif callable(getattr(value, name)):
                raise ValueError(f"Attribute {name} cannot be set by method of {str(value)}.")
            setter = (value, name)
            self.setters[name] = (ParameterSource.MODEL_ATTRIBUTE, setter, required)
            setattr(self, name, getattr(setter[0], setter[1]))
        elif isinstance(value, tuple) and len(value) >= 2 and isinstance(value[0], ParameterizedNode):
            # Case 3b: We are trying to access a parameter of a ParameterizedNode
            # with a different name.
            if not hasattr(value[0], value[1]):
                raise ValueError(f"Attribute {value[1]} missing from {str(value[0])}.")
            elif callable(getattr(value, name)):
                raise ValueError(f"Attribute {value[1]} cannot be set by method of {str(value[0])}.")
            self.setters[name] = (ParameterSource.MODEL_ATTRIBUTE, value, required)
            setattr(self, name, getattr(value[0], value[1]))
        else:
            # Case 4: The value is constant (including None).
            self.setters[name] = (ParameterSource.CONSTANT, value, required)
            setattr(self, name, value)

        if required and getattr(self, name) is None:
            raise ValueError(f"Missing required parameter {name}")

    def add_parameter(self, name, value=None, required=False, allow_overwrite=False, **kwargs):
        """Add a single *new* parameter to the ParameterizedNode.

        Notes
        -----
        * Checks multiple sources in the following order: Manually specified ``value``,
          an entry in ``kwargs``, or ``None``.
        * Sets an initial value for the attribute based on the given information.
        * The attributes are stored in the order in which they are added.

        Parameters
        ----------
        name : `str`
            The parameter name to add.
        value : any, optional
            The information to use to set the parameter. Can be a constant,
            function, ParameterizedNode, or self.
        required : `bool`
            Fail if the parameter is set to ``None``.
            Default = ``False``
        allow_overwrite : `bool`
            Allow a subclass to overwrite the definition of the attribute
            used in the superclass.
            Default = ``False``
        **kwargs : `dict`, optional
           All other keyword arguments, possibly including the parameter setters.

        Raises
        ------
        Raise a ``KeyError`` if there is a parameter collision or the parameter
        cannot be found.
        Raise a ``ValueError`` if the parameter is required, but set to None.
        """
        # Check for parameter collision.
        if hasattr(self, name) and getattr(self, name) is not None and not allow_overwrite:
            raise KeyError(f"Duplicate parameter set: {name}")

        # Add an entry for the setter function and fill in the remaining
        # information using set_parameter().
        self.setters[name] = (None, None, required)
        self.set_parameter(name, value, **kwargs)

    def sample_parameters(self, max_depth=50, **kwargs):
        """Sample the model's underlying parameters if they are provided by a function
        or ParameterizedNode.

        Parameters
        ----------
        max_depth : `int`
            The maximum recursive depth. Used to prevent infinite loops.
            Users should not need to set this manually.
        **kwargs : `dict`, optional
            All the keyword arguments, including the values needed to sample
            parameters.

        Raises
        ------
        Raise a ``ValueError`` the depth of the sampling encounters a problem
        with the order of dependencies.
        """
        if max_depth == 0:
            raise ValueError(f"Maximum sampling depth exceeded at {self}. Potential infinite loop.")

        # Increase the sampling iteration.
        self.sample_iteration += 1

        # Run through each parameter and sample it based on the given recipe.
        # As of Python 3.7 dictionaries are guaranteed to preserve insertion ordering,
        # so this will iterate through attributes in the order they were inserted.
        for name, (source_type, setter, _) in self.setters.items():
            sampled_value = None
            if source_type == ParameterSource.CONSTANT:
                sampled_value = setter
            elif source_type == ParameterSource.FUNCTION:
                sampled_value = setter(**kwargs)
            elif source_type == ParameterSource.MODEL_ATTRIBUTE:
                # Check if we need to resample the parent (before accessing the attribute).
                if setter.check_resample(self):
                    setter.sample_parameters(max_depth - 1, **kwargs)
                sampled_value = getattr(setter, name)
            elif source_type == ParameterSource.MODEL_METHOD:
                # Check if we need to resample the parent (before calling the method).
                parent_node = setter.__self__
                if parent_node.check_resample(self):
                    parent_node.sample_parameters(max_depth - 1, **kwargs)
                sampled_value = setter(**kwargs)
            else:
                raise ValueError(f"Unknown ParameterSource type {source_type} for {name}")
            setattr(self, name, sampled_value)

    def get_all_parameter_values(self, recursive=True, seen=None):
        """Get the values of the current parameters and (optionally) those of
        all their dependencies.

        Effectively snapshots the state of the execution graph.

        Parameters
        ----------
        seen : `set`
            A set of objects that have already been processed.
        recursive : `bool`
            Recursively extract the attribute setting of this object's dependencies.

        Returns
        -------
        values : `dict`
            The dictionary mapping the combination of the object identifier and
            attribute name to its value.
        """
        # Make sure that we do not process the same nodes multiple times.
        if seen is None:
            seen = set()
        if self in seen:
            return {}
        seen.add(self)

        values = {}
        for name, (source_type, setter, _) in self.setters.items():
            if recursive:
                if source_type == ParameterSource.MODEL_ATTRIBUTE:
                    values.update(setter.get_all_parameter_values(True, seen))
                elif source_type == ParameterSource.MODEL_METHOD:
                    values.update(setter.__self__.get_all_parameter_values(True, seen))

                full_name = f"{str(self)}.{name}"
            else:
                full_name = name
            values[full_name] = getattr(self, name)
        return values

    def get_dependencies(self, nodes=None):
        """Get all nodes on which this current node depends.

        Parameters
        ----------
        nodes : `set`
            A set of all nodes at or above this node in the graph.
            This is modified in place.

        Returns
        -------
        nodes : `set`
            A set of all nodes at or above this node in the graph.
        """
        # Make sure that we do not process the same nodes multiple times.
        if nodes is None:
            nodes = set()
        if self in nodes:
            return nodes
        nodes.add(self)

        for source_type, setter, _ in self.setters.values():
            if source_type == ParameterSource.MODEL_ATTRIBUTE:
                nodes = setter.get_dependencies(nodes)
            elif source_type == ParameterSource.MODEL_METHOD:
                nodes = setter.__self__.get_dependencies(nodes)
        return nodes


class FunctionNode(ParameterizedNode):
    """A class to wrap functions and their argument settings.

    Attributes
    ----------
    func : `function` or `method`
        The function to call during an evaluation.
    args_names : `list`
        A list of argument names to pass to the function.

    Parameters
    ----------
    func : `function` or `method`
        The function to call during an evaluation.
    node_identifier : `str`, optional
        An identifier (or name) for the current node.
    graph_base_seed : `int`, optional
        A base random seed to use for this specific evaluation graph.
        WARNING: This seed should almost never be set manually. Using the same
        seed for multiple graph instances will produce biased samples.
        If set to ``None`` will use urandom() to produce a fully random seed.
    **kwargs : `dict`, optional
        Any additional keyword arguments.

    Examples
    --------
    my_func = TDFunc(random.randint, a=1, b=10)
    value1 = my_func()      # Sample from default range
    value2 = my_func(b=20)  # Sample from extended range

    Note
    ----
    All the function's parameters that will be used need to be specified
    in either the default_args dict, object_args list, or as a kwarg in the
    constructor. Arguments cannot be first given during function call.
    For example the following will fail (because b is not defined in the
    constructor):

    my_func = TDFunc(random.randint, a=1)
    value1 = my_func(b=10.0)
    """

    def __init__(self, func, node_identifier=None, graph_base_seed=None, **kwargs):
        # We set the function before calling the parent class so we can use
        # the function's name (if needed).
        self.func = func
        super().__init__(node_identifier=node_identifier, graph_base_seed=graph_base_seed, **kwargs)

        # Add all of the parameters from default_args or the kwargs.
        self.arg_names = []
        for key, value in kwargs.items():
            self.arg_names.append(key)
            self.add_parameter(key, value)

    def __str__(self):
        """Return the string representation of the function."""
        # Extend the FunctionNode's string to include the name of the
        # function it calls so we can wrap a variety of raw functions.
        super_name = super().__str__()
        if self.func is not None:
            return f"{super_name}:{self.func.__name__}"
        else:
            return super_name

    def compute(self, **kwargs):
        """Execute the wrapped function.

        Parameters
        ----------
        **kwargs : `dict`, optional
            Additional function arguments.
        """
        args = {}
        for key in self.arg_names:
            # Override with the kwarg if the parameter is there.
            if key in kwargs:
                args[key] = kwargs[key]
            else:
                args[key] = getattr(self, key)
        return self.func(**args)
