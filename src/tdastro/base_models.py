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
5) The method of another ParameterizedNode.

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

import numpy as np


class ParameterSource(Enum):
    """ParameterSource specifies where a ParameterizedNode should get the value
    for a given parameter: a constant value or from another ParameterizedNode.
    """

    CONSTANT = 1
    MODEL_ATTRIBUTE = 2
    MODEL_METHOD = 3
    FUNCTION_NODE = 4


class ParameterizedNode:
    """Any model that uses parameters that can be set by constants,
    functions, or other parameterized nodes.

    Attributes
    ----------
    node_identifier : `str`
        An optional identifier (name) for the current node.
    setters : `dict` of `tuple`
        A dictionary to information about the setters for the parameters in the form:
        (ParameterSource, setter information, required). The attributes are
        stored in the order in which they need to be set.
    direct_dependencies : `dict`
        A dictionary with keys of other ParameterizedNodes on that this node needs to
        directly access. We use a dictionary to preserve ordering.
    _object_seed : `int` or None
        A object-specific seed to control random number generation.
    _graph_base_seed, `int` or None
        A base random seed to use for this specific evaluation graph. Used
        for validity checking.
    _node_id : `int` or None
        A unique ID number for each node in the graph. Assigned during
        resampling.

    Parameters
    ----------
    node_identifier : `str`, optional
        An identifier (or name) for the current node.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, node_identifier=None, **kwargs):
        self.setters = {}
        self.direct_dependencies = {}
        self.node_identifier = node_identifier
        self._node_id = None
        self._object_seed = None  # A default until set is called.
        self._graph_base_seed = None

        # We start all nodes with a completely random base seed.
        base_seed = int.from_bytes(urandom(4), "big")
        self.set_seed(graph_base_seed=base_seed)

    def __str__(self):
        """Return the string representation of the node."""
        name = f"{self.node_identifier}=" if self.node_identifier else ""
        id_str = f"{self._node_id}: " if self._node_id is not None else ""
        return f"{id_str}{name}{self.__class__.__module__}.{self.__class__.__qualname__}"

    def set_seed(self, new_seed=None, graph_base_seed=None):
        """Update the object seed to the new value based.

        The new value can be: 1) a given seed (new_seed), 2) a value computed from
        the graph's base seed (graph_base_seed) and the object's string representation,
        or a completely random seed (if neither option is set).

        WARNING: This seed should almost never be set manually. Using the duplicate
        seeds for multiple graph instances or runs will produce biased samples.

        Parameters
        ----------
        new_seed : `int`, optional
            The given seed
        graph_base_seed : `int`, optional
            A base random seed to use for this specific evaluation graph.
        """
        # If we are given a predefined seed, use that.
        if new_seed is not None:
            self._object_seed = new_seed
            return

        # Update the base seed if needed.
        if graph_base_seed is not None:
            self._graph_base_seed = graph_base_seed

        hashed_object_name = md5(str(self).encode())
        seed_offset = int(hashed_object_name.hexdigest(), base=16)
        new_seed = (self._graph_base_seed + seed_offset) % (2**32)
        self._object_seed = new_seed

    def update_graph_information(self, new_graph_base_seed=None, seen_nodes=None):
        """Force an update of the graph structure and the seeds.

        Updates the node ids to capture their location in the graph. Also updates
        the graph_base_seed (if given).

        WARNING: This will modify the per-node seeds to account for the new graph base
        seed (if there is one) and any changes in their position within the graph structure.

        Parameters
        ----------
        new_graph_base_seed : `int`, optional
            A base random seed to use for this specific evaluation graph.
        seen_nodes : `set`, optional
            A set of nodes that have already been processed to prevent infinite loops.
            Caller should not set.
        """
        # Make sure that we do not process the same nodes multiple times.
        if seen_nodes is None:
            seen_nodes = set()
        if self in seen_nodes:
            return
        seen_nodes.add(self)

        # Update the graph ID and (possibly) the seed.
        self._node_id = len(seen_nodes) - 1
        self.set_seed(graph_base_seed=new_graph_base_seed)

        # Recursively update any direct dependencies.
        for dep in self.direct_dependencies:
            dep.update_graph_information(new_graph_base_seed, seen_nodes)

    def _update_dependencies(self):
        """Update the set of direct dependencies."""
        self.direct_dependencies = {}
        for source_type, setter, _ in self.setters.values():
            current = None
            if source_type == ParameterSource.MODEL_ATTRIBUTE:
                current = setter[0]
            elif source_type == ParameterSource.MODEL_METHOD:
                current = setter.__self__
            elif source_type == ParameterSource.FUNCTION_NODE:
                current = setter

            if current is not None and current is not self:
                self.direct_dependencies[current] = True

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

        sampled_value = None
        if callable(value):
            if isinstance(value, types.FunctionType):
                # Case 1a: This is a static function (not attached to an object).
                # Wrap the function in a FunctionNode.
                func_node = FunctionNode(value, **kwargs)
                self.setters[name] = (ParameterSource.FUNCTION_NODE, func_node, required)
                sampled_value = func_node.compute()
            elif isinstance(value.__self__, ParameterizedNode):
                # Case 1b: This is a method attached to another ParameterizedNode.
                self.setters[name] = (ParameterSource.MODEL_METHOD, value, required)
                sampled_value = value(**kwargs)
            else:
                # Case 1c: This is a general callable method from another object.
                # We treat it as static (we don't resample the other object) and
                # wrap it in a FunctionNode.
                func_node = FunctionNode(value, **kwargs)
                self.setters[name] = (ParameterSource.FUNCTION_NODE, func_node, required)
                sampled_value = func_node.compute()
        elif isinstance(value, FunctionNode):
            # Case 2: We are using the result of a computation of the function node.
            self.setters[name] = (ParameterSource.FUNCTION_NODE, value, required)
            sampled_value = value.compute(**kwargs)
        elif isinstance(value, ParameterizedNode):
            # Case 3a: We are trying to access a parameter of a ParameterizedNode
            # with the same name.
            if value == self:
                raise ValueError(f"Attribute {name} is recursively assigned to self.{name}.")
            if not hasattr(value, name):
                raise ValueError(f"Attribute {name} missing from {str(value)}.")
            if callable(getattr(value, name)):
                raise ValueError(f"{value}.{name} is callable and should be an attribute.")

            # We store MODEL_ATTRIBUTE setters as a tuple of (object, attribute_name).
            setter = (value, name)
            self.setters[name] = (ParameterSource.MODEL_ATTRIBUTE, setter, required)
            sampled_value = getattr(setter[0], setter[1])
        elif isinstance(value, tuple) and len(value) >= 2 and isinstance(value[0], ParameterizedNode):
            # Case 3b: We are trying to access a parameter of a ParameterizedNode
            # with a different name.
            if not hasattr(value[0], value[1]):
                raise ValueError(f"Attribute {value[1]} missing from {str(value[0])}.")
            elif callable(getattr(value[0], value[1])):
                raise ValueError(f"{value[0]}.{value[1]} is callable and should be an attribute.")
            self.setters[name] = (ParameterSource.MODEL_ATTRIBUTE, value, required)
            sampled_value = getattr(value[0], value[1])
        else:
            # Case 4: The value is constant (including None).
            self.setters[name] = (ParameterSource.CONSTANT, value, required)
            sampled_value = value

        # Check that we did get a parameter.
        if required and sampled_value is None:
            raise ValueError(f"Missing required parameter {name}")
        setattr(self, name, sampled_value)

        # Update the dependencies to account for any new nodes in the graph.
        self._update_dependencies()

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

    def _sample_helper(self, depth, seen_nodes, **kwargs):
        """Internal recursive function to sample the model's underlying parameters
        if they are provided by a function or ParameterizedNode.

        Parameters
        ----------
        depth : `int`
            The recursive depth remaining. Used to prevent infinite loops.
            Users should not need to set this manually.
        seen_nodes : `dict`
            A dictionary mapping nodes seen during this sampling run to their ID.
            Used to avoid sampling nodes multiple times and to validity check the graph.
        **kwargs : `dict`, optional
            All the keyword arguments, including the values needed to sample
            parameters.

        Raises
        ------
        Raise a ``ValueError`` the depth of the sampling encounters a problem
        with the order of dependencies.
        """
        if depth == 0:
            raise ValueError(f"Maximum sampling depth exceeded at {self}. Potential infinite loop.")
        if self in seen_nodes:
            return  # Nothing to do
        seen_nodes[self] = self._node_id

        # Run through each parameter and sample it based on the given recipe.
        # As of Python 3.7 dictionaries are guaranteed to preserve insertion ordering,
        # so this will iterate through attributes in the order they were inserted.
        for name, (source_type, setter, _) in self.setters.items():
            sampled_value = None
            if source_type == ParameterSource.CONSTANT:
                sampled_value = setter
            elif source_type == ParameterSource.MODEL_ATTRIBUTE:
                # Check if we need to resample the parent (before accessing the attribute).
                if setter[0] not in seen_nodes:
                    setter[0]._sample_helper(depth - 1, seen_nodes, **kwargs)
                sampled_value = getattr(setter[0], setter[1])
            elif source_type == ParameterSource.MODEL_METHOD:
                # Check if we need to resample the parent (before calling the method).
                parent_node = setter.__self__
                if parent_node not in seen_nodes:
                    parent_node._sample_helper(depth - 1, seen_nodes, **kwargs)
                sampled_value = setter(**kwargs)
            elif source_type == ParameterSource.FUNCTION_NODE:
                # Check if we need to resample the parent function (before calling compute).
                if setter not in seen_nodes:
                    setter._sample_helper(depth - 1, seen_nodes, **kwargs)
                sampled_value = setter.compute(**kwargs)
            else:
                raise ValueError(f"Unknown ParameterSource type {source_type} for {name}")
            setattr(self, name, sampled_value)

    def sample_parameters(self, **kwargs):
        """Sample the model's underlying parameters if they are provided by a function
        or ParameterizedNode.

        Parameters
        ----------
        **kwargs : `dict`, optional
            All the keyword arguments, including the values needed to sample
            parameters.

        Raises
        ------
        Raise a ``ValueError`` the depth of the sampling encounters a problem
        with the order of dependencies.
        """
        # If the graph structure has never been set, do that now.
        if self._node_id is None:
            nodes = set()
            self.update_graph_information(seen_nodes=nodes)

        # Resample the nodes.
        seen_nodes = {}
        self._sample_helper(50, seen_nodes, **kwargs)

        # Validity check that we do not see duplicate IDs.
        seen_ids = np.array(list(seen_nodes.values()))
        if len(seen_ids) != len(np.unique(seen_ids)):
            raise ValueError(
                "The graph nodes do not have unique IDs, which can indicate the graph "
                "structure has changed. Please use update_graph_information()."
            )

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
        # If we haven't processed the nodes yet, do that.
        if self._node_id is None:
            nodes = set()
            self.update_graph_information(seen_nodes=nodes)

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
                    values.update(setter[0].get_all_parameter_values(True, seen))
                elif source_type == ParameterSource.MODEL_METHOD:
                    values.update(setter.__self__.get_all_parameter_values(True, seen))
                elif source_type == ParameterSource.FUNCTION_NODE:
                    values.update(setter.get_all_parameter_values(True, seen))

                full_name = f"{str(self)}.{name}"
            else:
                full_name = name
            values[full_name] = getattr(self, name)
        return values


class SingleVariableNode(ParameterizedNode):
    """A ParameterizedNode holding a single pre-defined variable.

    Notes
    -----
    Often used for testing, but can be used to make graph dependencies clearer.

    Parameters
    ----------
    name : `str`
        The parameter name.
    value : any
        The parameter value.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, name, value, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter(name, value, required=True, **kwargs)


class FunctionNode(ParameterizedNode):
    """A class to wrap functions and their argument settings.

    The node can compute the result using a given function (the ``func``
    parameter) or through the ``compute()`` method. If no ``func=None``
    then the user must override ``compute()``.

    Attributes
    ----------
    func : `function` or `method`
        The function to call during an evaluation. If this is ``None``
        you must override the ``compute()`` method directly.
    args_names : `list`
        A list of argument names to pass to the function.
    outputs : `list` of `str`
        The output attributes of this function.
    num_outputs : `int`
        The number of outputs.

    Parameters
    ----------
    func : `function` or `method`
        The function to call during an evaluation.
    node_identifier : `str`, optional
        An identifier (or name) for the current node.
    outputs : `list` of `str`, optional
        The output attributes of this function. If ``None`` uses
        a single attribute ``result``.
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

    def __init__(self, func, node_identifier=None, outputs=None, **kwargs):
        # We set the function before calling the parent class so we can use
        # the function's name (if needed).
        self.func = func
        super().__init__(node_identifier=node_identifier, **kwargs)

        # Add all of the parameters from default_args or the kwargs.
        self.arg_names = []
        for key, value in kwargs.items():
            self.arg_names.append(key)
            self.add_parameter(key, value)

        # Add the output arguments.
        if not outputs:
            outputs = ["result"]
        self.outputs = outputs
        self.num_outputs = len(outputs)
        for name in outputs:
            self.add_parameter(name, None)

    def __str__(self):
        """Return the string representation of the function."""
        # Extend the FunctionNode's string to include the name of the
        # function it calls so we can wrap a variety of raw functions.
        super_name = super().__str__()
        if self.func is None:
            return super_name
        return f"{super_name}:{self.func.__name__}"

    def _build_args_dict(self, **kwargs):
        """Build a dictionary of arguments for the function."""
        args = {}
        for key in self.arg_names:
            # Override with the kwarg if the parameter is there.
            if key in kwargs:
                args[key] = kwargs[key]
            else:
                args[key] = getattr(self, key)
        return args

    def _sample_helper(self, depth, seen_nodes, **kwargs):
        """Internal recursive function to sample the model's underlying parameters
        if they are provided by a function or ParameterizedNode.

        Parameters
        ----------
        depth : `int`
            The recursive depth remaining. Used to prevent infinite loops.
            Users should not need to set this manually.
        seen_nodes : `dict`
            A dictionary mapping nodes seen during this sampling run to their ID.
            Used to avoid sampling nodes multiple times and to validity check the graph.
        **kwargs : `dict`, optional
            All the keyword arguments, including the values needed to sample
            parameters.

        Raises
        ------
        Raise a ``ValueError`` the depth of the sampling encounters a problem
        with the order of dependencies.
        """
        super()._sample_helper(depth, seen_nodes, **kwargs)
        _ = self.compute(**kwargs)

    def compute(self, **kwargs):
        """Execute the wrapped function.

        Parameters
        ----------
        **kwargs : `dict`, optional
            Additional function arguments.

        Raises
        ------
        ``ValueError`` is ``func`` attribute is ``None``.
        """
        if self.func is None:
            raise ValueError(
                "func parameter is None for a FunctionNode. You need to either "
                "set func or override compute()."
            )

        args = self._build_args_dict(**kwargs)
        results = self.func(**args)
        if self.num_outputs == 1:
            setattr(self, f"{self.outputs[0]}", results)
        else:
            if len(results) != self.num_outputs:
                raise ValueError(
                    f"Incorrect number of results returned by {self.func.__name__}. "
                    f"Expected {self.outputs}, but got {results}."
                )
            for i in range(self.num_outputs):
                setattr(self, f"{self.outputs[i]}", results[i])

        return results
