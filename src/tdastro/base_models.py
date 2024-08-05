"""The base models used to specify the TDAstro computation graph.

The computation graph is composed of ParameterizedNodes which store random variables
(called "model parameters" or "parameters" for short) and encode the dependencies between
them. Model parameters are different from object variables in that they are programmatically
set by sampling from the graph. They are stored in a special parameters dictionary and have
limited write access.

All model parameters (random variables in the probabilistic graph) must be added using
using the ParameterizedNode.add_parameter() function. This allows the graph to track
which parameters to update and how to set them. Parameter's values can be set from a few
sources:
1) A constant (Example: A given standard deviation for a noise model)
2) A static function or method (which does not have variables that are resampled).
3) The result of evaluating a FunctionNode, which provides a computation using other
   parameters in the graph.
4) The parameters of another ParameterizedNode.

We say that parameter X is dependent on parameter Y if the value of Y is necessary
to compute the value of X. For example if X is set by evaluating a FunctionNode that
uses parameter Y in the computation, X is dependent on Y. The dependencies impose an
ordering of model parameters in the graph. Y must be computed before X.

ParameterNodes provide semantic groupings of individual parameters. For example we may
have a ParameterNode representing the information needed for a Type Ia supernova.
That node's parameters would include the variables needed to evaluate the supernova's
lightcurve. Each of these parameters might depend on parameters in other nodes, such
as those of the host galaxy.

The execution graph is processed by starting at the final node, examining each
model parameter for that node, and recursively proceeding 'up' the graph for any
of its parameters that has a dependency. For example the function.

f(a, b) = x
g(c) = y
h(x, y) = z

would form the graph:

a -\
    x - \
b -/     \
          z
c -- y -- /

where z is the 'bottom' node. Parameters a, b, and c would be at the 'top' of the
graph because they have no dependencies.  Such parameters are set by constants or
static functions.
"""

from hashlib import md5
from os import urandom

import numpy as np


class ParameterSource:
    """ParameterSource specifies the information about where a ParameterizedNode should
    get the value for a given parameter.

    Attributes
    ----------
    parameter_name : `str`
        The name of the parameter within the node (short name).
    source_type : `int`
        The type of source as defined by the class variables.
        Default = 0
    value : any
        The information that actually sets the parameter. Either a constant
        or the attribute name of a dependency node.
    dependency : `ParameterizedNode` or None
        The node on which this parameter is dependent
    fixed : `bool`
        The attribute cannot be changed during resampling.
        Default = ``False``
    required : `bool`
        The attribute must exist and be non-None.
        Default = ``False``
    full_name : `str`
        The full name of the parameter including the node information.
    """

    # Class variables for the source enum.
    UNDEFINED = 0
    CONSTANT = 1
    MODEL_PARAMETER = 2
    FUNCTION_NODE = 3

    def __init__(self, parameter_name, source_type=0, fixed=False, required=False, node_name=""):
        self.source_type = source_type
        self.fixed = fixed
        self.required = required
        self.value = None
        self.dependency = None
        self.set_name(parameter_name, node_name)

    def get_value(self, **kwargs):
        """Get the parameter's value."""
        if self.source_type == ParameterSource.CONSTANT:
            return self.value
        elif self.source_type == ParameterSource.MODEL_PARAMETER:
            return self.dependency.parameters[self.value]
        elif self.source_type == ParameterSource.FUNCTION_NODE:
            return self.dependency.parameters[self.value]
        else:
            raise ValueError(f"Invalid ParameterSource type {self.source_type}")

    def set_name(self, parameter_name="", node_name=""):
        """Set the name of the parameter field.

        Parameter
        ---------
        parameter_name : `str`
            The name of the parameter within the node (short name).
        node_name : `str`
            The node string for the node containing this parameter.
        """
        if len(parameter_name) == 0:
            raise ValueError(f"Invalid parameter name: {parameter_name}")

        self.parameter_name = parameter_name
        if len(node_name) > 0:
            self.full_name = f"{node_name}.{parameter_name}"
        else:
            self.full_name = f"{parameter_name}"

    def set_as_constant(self, value):
        """Set the parameter as a constant value.

        Parameters
        ----------
        value : any
            The constant value to use.
        """
        if callable(value):
            raise ValueError(f"Using set_as_constant on callable {value}")

        self.source_type = ParameterSource.CONSTANT
        self.dependency = None
        self.value = value

    def set_as_parameter(self, dependency, param_name):
        """Set the parameter as a model parameter of another node.

        Parameters
        ----------
        dependency : `ParameterizedNode`
            The node in which to access the attribute.
        param_name : `str`
            The name of the parameter to access.
        """
        self.source_type = ParameterSource.MODEL_PARAMETER
        self.dependency = dependency
        self.value = param_name

    def set_as_function(self, dependency, param_name="function_node_result"):
        """Set the parameter as a model parameter of another node.

        Parameters
        ----------
        dependency : `ParameterizedNode`
            The node in which to access the attribute.
        param_name : `str`
            The name of where the result is stored in the FunctionNode.
        """
        self.source_type = ParameterSource.FUNCTION_NODE
        self.dependency = dependency
        self.value = param_name


class ParameterizedNode:
    """Any model that uses parameters that can be set by constants,
    functions, or other parameterized nodes.

    Attributes
    ----------
    node_label : `str`
        An optional human readable identifier (name) for the current node.
    node_string : `str`
        The full string used to identify a node. This is a combination of the nodes position
        in the graph (if known), node_label (if provided), and class information.
    setters : `dict`
        A dictionary mapping the parameters' names to information about the setters
        (ParameterSource). The model parameters are stored in the order in which they
        need to be set.
    parameters : `dict`
        A dictionary mapping the parameter's name to its current value.
    direct_dependencies : `dict`
        A dictionary with keys of other ParameterizedNodes on that this node needs to
        directly access. We use a dictionary to preserve ordering.
    _object_seed : `int` or None
        A object-specific seed to control random number generation.
    _graph_base_seed, `int` or None
        A base random seed to use for this specific evaluation graph. Used
        for validity checking.
    _node_pos : `int` or None
        A unique ID number for each node in the graph indicating its position.
        Assigned during resampling or `update_graph_information()`

    Parameters
    ----------
    node_label : `str`, optional
        An identifier (or name) for the current node.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, node_label=None, **kwargs):
        self.setters = {}
        self.parameters = {}
        self.direct_dependencies = {}
        self.node_label = node_label
        self._node_pos = None
        self._object_seed = None  # A default until set is called.
        self._graph_base_seed = None
        self.node_string = None

        # We start all nodes with a completely random base seed.
        base_seed = int.from_bytes(urandom(4), "big")
        self.set_seed(graph_base_seed=base_seed)

    def __str__(self):
        """Return the string representation of the node."""
        if self.node_string is None:
            # Create and cache the node string.
            self._update_node_string()
        return self.node_string

    def _update_node_string(self, extra_tag=None):
        """Update the node's string."""
        pos_string = f"{self._node_pos}:" if self._node_pos is not None else ""
        if self.node_label is not None:
            self.node_string = f"{pos_string}{self.node_label}"
        else:
            self.node_string = f"{pos_string}{self.__class__.__module__}.{self.__class__.__qualname__}"

        # Allow for the appending of an extra tag.
        if extra_tag is not None:
            self.node_string = f"{self.node_string}:{extra_tag}"

        # Update the full_name of all node's parameter setters.
        for name, setter_info in self.setters.items():
            setter_info.set_name(name, self.node_string)

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

        # Force an update of the node string to make sure we have the most recent.
        self._update_node_string()
        hashed_object_name = md5(self.node_string.encode())

        seed_offset = int(hashed_object_name.hexdigest(), base=16)
        new_seed = (self._graph_base_seed + seed_offset) % (2**32)
        self._object_seed = new_seed

    def update_graph_information(
        self,
        new_graph_base_seed=None,
        seen_nodes=None,
        reset_variables=False,
    ):
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
        reset_variables : `bool`
            Force all variables to ``None`` so that the code will check their
            dependency order when resampling.
        """
        # Make sure that we do not process the same nodes multiple times.
        if seen_nodes is None:
            seen_nodes = set()
        if self in seen_nodes:
            return
        seen_nodes.add(self)

        # Update the graph ID and (possibly) the seed.
        self._node_pos = len(seen_nodes) - 1
        self._update_node_string()
        self.set_seed(graph_base_seed=new_graph_base_seed)

        # Reset the variables if needed.
        if reset_variables:
            for name in self.setters:
                self.parameters[name] = None

        # Recursively update any direct dependencies.
        for dep in self.direct_dependencies:
            dep.update_graph_information(new_graph_base_seed, seen_nodes, reset_variables)

    def __getitem__(self, key):
        return self.parameters[key]

    def set_parameter(self, name, value=None, **kwargs):
        """Set a single *existing* parameter to the ParameterizedNode.

        Notes
        -----
        * Sets an initial value for the model parameter based on the given information.
        * The model parameters are stored in the order in which they are added.

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
        # Check for parameter has been added and if so, find the index. All parameters must
        # be added first with "add_parameter()".
        if name not in self.setters:
            raise KeyError(f"Tried to set parameter {name} that has not been added.") from None

        if value is None and name in kwargs:
            # The value wasn't set, but the name is in kwargs.
            value = kwargs[name]

        if callable(value):
            if "__self__" in value.__dir__() and isinstance(value.__self__, ParameterizedNode):
                # Case 1a: This is a method attached to another ParameterizedNode.
                # Check if this is a getter method. If so, we access the parameter directly
                # to save a function call.
                method_name = value.__name__
                parent = value.__self__
                if method_name in parent.parameters:
                    self.setters[name].set_as_parameter(parent, method_name)
                else:
                    raise ValueError(f"Trying to set parameter to method {method_name}")
            else:
                # Case 1b: This is a general function or callable method from another object.
                # We treat it as static (we don't resample the other object) and
                # wrap it in a FunctionNode.
                func_node = FunctionNode(value, **kwargs)
                self.setters[name].set_as_function(func_node)
        elif isinstance(value, FunctionNode):
            # Case 2: We are using the result of a computation of the function node.
            self.setters[name].set_as_function(value)
        elif isinstance(value, ParameterizedNode):
            # Case 3: We are trying to access a parameter of a ParameterizedNode
            # with the same name.
            if value == self:
                raise ValueError(f"Parameter {name} is recursively assigned to self.{name}.")
            if name not in value.parameters:
                raise ValueError(f"Parameter {name} missing from {str(value)}.")
            self.setters[name].set_as_parameter(value, name)
        else:
            # Case 4: The value is constant (including None).
            self.setters[name].set_as_constant(value)

        # Check that we did get a parameter.
        sampled_value = self.setters[name].get_value(**kwargs)
        if self.setters[name].required and sampled_value is None:
            raise ValueError(f"Missing required parameter {name}")
        self.parameters[name] = sampled_value

        # Update the dependencies to account for any new nodes in the graph.
        self.direct_dependencies = {}
        for setter_info in self.setters.values():
            if setter_info.dependency is not None and setter_info.dependency is not self:
                self.direct_dependencies[setter_info.dependency] = True

    def add_parameter(self, name, value=None, required=False, fixed=False, **kwargs):
        """Add a single *new* parameter to the ParameterizedNode.

        Notes
        -----
        * Checks multiple sources in the following order: Manually specified ``value``,
          an entry in ``kwargs``, or ``None``.
        * Sets an initial value for the model parameter based on the given information.
        * The model parameters are stored in the order in which they are added.

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
        fixed : `bool`
            The attribute cannot be changed during resampling.
            Default = ``False``
        **kwargs : `dict`, optional
           All other keyword arguments, possibly including the parameter setters.

        Raises
        ------
        Raise a ``KeyError`` if there is a parameter collision or the parameter
        cannot be found.
        Raise a ``ValueError`` if the parameter is required, but set to None.
        """
        # Check for parameter collision and add a place holder value.
        if hasattr(self, name) and name not in self.parameters:
            raise KeyError(f"Parameter name {name} conflicts with a predefined model parameter.")
        if self.parameters.get(name, None) is not None:
            raise KeyError(f"Duplicate parameter set: {name}")
        self.parameters[name] = None

        # Add an entry for the setter function and fill in the remaining information using
        # set_parameter(). We add an initial (dummy) value here to indicate that this parameter
        # exists and was added via add_parameter().
        self.setters[name] = ParameterSource(
            parameter_name=name,
            source_type=ParameterSource.UNDEFINED,
            fixed=fixed,
            required=required,
            node_name=str(self),
        )
        self.set_parameter(name, value, **kwargs)

        # Create a callable getter function using. We override the __self__ and __name__
        # attributes so it looks like method of this object.
        # This allows us to reference the parameter as object.parameter_name for chaining.
        def getter():
            return self.parameters[name]

        getter.__self__ = self
        getter.__name__ = name
        setattr(self, name, getter)

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
        seen_nodes[self] = self._node_pos

        # Run through each parameter and sample it based on the given recipe.
        # As of Python 3.7 dictionaries are guaranteed to preserve insertion ordering,
        # so this will iterate through model parameters in the order they were inserted.
        for name, setter_info in self.setters.items():
            if setter_info.dependency is not None:
                setter_info.dependency._sample_helper(depth - 1, seen_nodes, **kwargs)
            self.parameters[name] = setter_info.get_value(**kwargs)

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
        if self._node_pos is None:
            nodes = set()
            self.update_graph_information(seen_nodes=nodes, reset_variables=True)

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
            Recursively extract the model parameter settings of this object's dependencies.

        Returns
        -------
        values : `dict`
            The dictionary mapping the combination of the object identifier and
            model parameter name to its value.
        """
        # If we haven't processed the nodes yet, do that.
        if self._node_pos is None:
            nodes = set()
            self.update_graph_information(seen_nodes=nodes)

        # Make sure that we do not process the same nodes multiple times.
        if seen is None:
            seen = set()
        if self in seen:
            return {}
        seen.add(self)

        all_values = {}
        for name, setter_info in self.setters.items():
            if recursive:
                if setter_info.dependency is not None:
                    all_values.update(setter_info.dependency.get_all_parameter_values(True, seen))
                full_name = setter_info.full_name
            else:
                full_name = name
            all_values[full_name] = self.parameters[name]
        return all_values


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
        The output model parameters of this function.

    Parameters
    ----------
    func : `function` or `method`
        The function to call during an evaluation.
    node_label : `str`, optional
        An identifier (or name) for the current node.
    outputs : `list` of `str`, optional
        The output model parameters of this function. If ``None`` uses
        a single model parameter ``result``.
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

    def __init__(self, func, node_label=None, outputs=None, **kwargs):
        # We set the function before calling the parent class so we can use
        # the function's name (if needed).
        self.func = func
        super().__init__(node_label=node_label, **kwargs)

        # Add all of the parameters from default_args or the kwargs.
        self.arg_names = []
        for key, value in kwargs.items():
            self.arg_names.append(key)
            self.add_parameter(key, value)

        # Add the output arguments.
        if not outputs:
            outputs = ["function_node_result"]
        self.outputs = outputs
        for name in outputs:
            # For output parameters we add a placeholder of None to set up the basic data, such as
            # the getter function and the entry in parameters. Then we change the
            # type to point to own result.
            self.add_parameter(name, None)
            self.setters[name].set_as_function(self, param_name=name)

        # Fill in the outputs.
        self.compute()

    def _update_node_string(self, extra_tag=None):
        """Update the node's string."""
        if extra_tag is not None:
            super()._update_node_string(extra_tag)
        elif self.func is not None:
            super()._update_node_string(self.func.__name__)
        else:
            super()._update_node_string()

    def _build_args_dict(self, **kwargs):
        """Build a dictionary of arguments for the function."""
        args = {}
        for key in self.arg_names:
            # Override with the kwarg if the parameter is there.
            if key in kwargs:
                args[key] = kwargs[key]
            else:
                args[key] = self.parameters[key]
        return args

    def _save_result_parameters(self, results):
        """Save the results as model parameters.

        Parameters
        ----------
        results : any
            The results of the function.
        """
        if len(self.outputs) == 1:
            self.parameters[self.outputs[0]] = results
        else:
            if len(results) != len(self.outputs):
                raise ValueError(
                    f"Incorrect number of results returned by {self.func.__name__}. "
                    f"Expected {len(self.outputs)}, but got {results}."
                )
            for i in range(len(self.outputs)):
                self.parameters[self.outputs[i]] = results[i]

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
        if self not in seen_nodes:
            # First use _sample_helper() to update the function node's inputs (dependencies).
            # Then use compute() to update the function node's outputs.
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
        self._save_result_parameters(results)
        return results
