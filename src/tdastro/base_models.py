"""The base models used to specify the TDAstro computation graph.

The computation graph is composed of ParameterizedNodes which compute random variables
(called "model parameters" or "parameters" for short) and encode the dependencies between
them. Model parameters are different from object variables in that they are *not* stored
within the object, but rather programmatically set in an external graph_state dictionary by
sampling the graph. The user does not need to access the internals of graph_state directly,
but rather can treat it as an opaque object to pass around. The dictionary can hold either
individual values (often floats) or arrays of samples.

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

from tdastro.graph_state import GraphState


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
    COMPUTE_OUTPUT = 4

    def __init__(self, parameter_name, source_type=0, fixed=False, required=False, node_name=""):
        self.source_type = source_type
        self.fixed = fixed
        self.required = required
        self.value = None
        self.dependency = None
        self.set_name(parameter_name, node_name)

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

    def set_as_compute_output(self, param_name="function_node_result"):
        """Set the parameter the output of this current node's compute() method.

        Parameters
        ----------
        dependency : `ParameterizedNode`
            The node in which to access the attribute.
        param_name : `str`
            The name of where the result is stored in the FunctionNode.
        """
        self.source_type = ParameterSource.COMPUTE_OUTPUT
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
    node_hash : `int`
        A hashed version of ``node_string`` used for fast lookups.
    setters : `dict`
        A dictionary mapping the parameters' names to information about the setters
        (ParameterSource). The model parameters are stored in the order in which they
        need to be set.
    direct_dependencies : `dict`
        A dictionary with keys of other ParameterizedNodes on that this node needs to
        directly access. We use a dictionary to preserve ordering.
    node_pos : `int` or None
        A unique ID number for each node in the graph indicating its position.
        Assigned during resampling or `set_graph_positions()`

    Parameters
    ----------
    node_label : `str`, optional
        An identifier (or name) for the current node.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, node_label=None, **kwargs):
        self.setters = {}
        self.direct_dependencies = {}
        self.node_label = node_label
        self.node_pos = None
        self.node_string = None
        self.node_hash = None

    def __str__(self):
        """Return the string representation of the node."""
        if self.node_string is None:
            # Create and cache the node string.
            self._update_node_string()
        return self.node_string

    def _update_node_string(self, extra_tag=None):
        """Update the node's string."""
        pos_string = f"{self.node_pos}:" if self.node_pos is not None else ""
        if self.node_label is not None:
            self.node_string = f"{pos_string}{self.node_label}"
        else:
            self.node_string = f"{pos_string}{self.__class__.__module__}.{self.__class__.__qualname__}"

        # Allow for the appending of an extra tag.
        if extra_tag is not None:
            self.node_string = f"{self.node_string}:{extra_tag}"

        # Save the hashed value of the node string.
        hashed_object_name = md5(self.node_string.encode()).hexdigest()
        self.node_hash = int(hashed_object_name, base=16)

        # Update the full_name of all node's parameter setters.
        for name, setter_info in self.setters.items():
            setter_info.set_name(name, self.node_string)

    def set_graph_positions(self, seen_nodes=None):
        """Force an update of the graph structure (numbering of each node).

        Parameters
        ----------
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

        # Update the node's position in the graph and its string.
        self.node_pos = len(seen_nodes) - 1
        self._update_node_string()

        # Recursively update any direct dependencies.
        for dep in self.direct_dependencies:
            dep.set_graph_positions(seen_nodes)

    def get_param(self, graph_state, name):
        """Get the value of a parameter stored in this node.

        Note
        ----
        This is an optional helper function that accesses the internals of graph_state
        using the node's information (e.g. its hash value).

        Parameters
        ----------
        graph_state : `dict`
            The dictionary of graph state information.
        name : `str`
            The parameter name to query.

        Returns
        -------
        any
            The parameter value.

        Raises
        ------
        ``KeyError`` if this parameter has not be set.
        ``ValueError`` if graph_state is None.
        """
        if graph_state is None:
            raise ValueError(f"Unable to look ip parameter={name}. No graph_state given.")
        return graph_state[self.node_hash][name]

    def get_local_params(self, graph_state):
        """Get a dictionary of all parameters local to this node.

        Note
        ----
        This is an optional helper function that accesses the internals of graph_state
        using the node's information (e.g. its hash value).

        Parameters
        ----------
        graph_state : `GraphState`
            An object mapping graph parameters to their values.

        Returns
        -------
        result : `dict`
            A dictionary mapping the parameter name to its value.

        Raises
        ------
        ``KeyError`` if no parameters have been set for this node.
        ``ValueError`` if graph_state is None.
        """
        if graph_state is None:
            raise ValueError("No graph_state given.")
        return graph_state[self.node_hash]

    def set_parameter(self, name, value=None, **kwargs):
        """Set a single *existing* parameter to the ParameterizedNode.

        Notes
        -----
        * Does NOT set an initial value for the model parameter. The user must
          sample the parameters for this to be set.
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
        # Set the node's position in the graph to None to indicate that the
        # structure might have changed. It needs to be updated with set_graph_positions().
        self.node_pos = None

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
                if method_name in parent.setters:
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
            if name not in value.setters:
                raise ValueError(f"Parameter {name} missing from {str(value)}.")
            self.setters[name].set_as_parameter(value, name)
        else:
            # Case 4: The value is constant (including None).
            self.setters[name].set_as_constant(value)
            if self.setters[name].required and value is None:
                raise ValueError(f"Missing required parameter {name}")

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
        * Does NOT set an initial value for the model parameter. The user must
          sample the parameters for this to be set.
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
        if hasattr(self, name) and name not in self.setters:
            raise KeyError(f"Parameter name {name} conflicts with a predefined model parameter.")
        if self.setters.get(name, None) is not None:
            raise KeyError(f"Duplicate parameter set: {name}")

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

        # Only constant sources can be fixed.
        if fixed and self.setters[name].source_type != ParameterSource.CONSTANT:
            raise ValueError(f"Tried to make {name} fixed but source_type={self.setters[name].source_type}.")

        # Create a callable getter function using. We override the __self__ and __name__
        # attributes so it looks like method of this object.
        # This allows us to reference the parameter as object.parameter_name for chaining.
        def getter():
            return None

        getter.__self__ = self
        getter.__name__ = name
        setattr(self, name, getter)

    def compute(self, graph_state, given_args=None, rng_info=None, **kwargs):
        """Placeholder for a general compute function.

        Parameters
        ----------
        graph_state : `GraphState`
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        rng_info : `dict`, optional
            A dictionary of random number generator information for each node, such as
            the JAX keys or the numpy rngs.
        **kwargs : `dict`, optional
            Additional function arguments.
        """
        return None

    def _sample_helper(self, graph_state, seen_nodes, given_args=None, rng_info=None):
        """Internal recursive function to sample the model's underlying parameters
        if they are provided by a function or ParameterizedNode. All sampled
        parameters for all nodes are stored in the graph_state dictionary, which is
        modified in-place.

        Parameters
        ----------
        graph_state : `GraphState`
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        seen_nodes : `dict`
            A dictionary mapping nodes seen during this sampling run to their ID.
            Used to avoid sampling nodes multiple times and to validity check the graph.
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        rng_info : `dict`, optional
            A dictionary of random number generator information for each node, such as
            the JAX keys or the numpy rngs.

        Raises
        ------
        Raise a ``KeyError`` if the sampling encounters an error with the order of dependencies.
        """
        if self in seen_nodes:
            return  # Nothing to do
        seen_nodes[self] = self.node_pos

        # Run through each parameter and sample it based on the given recipe.
        # As of Python 3.7 dictionaries are guaranteed to preserve insertion ordering,
        # so this will iterate through model parameters in the order they were inserted.
        any_compute = False
        for name, setter in self.setters.items():
            # If we are given the argument use that and do not worry about the dependencies.
            if given_args is not None and setter.full_name in given_args:
                if setter.fixed:
                    raise ValueError(f"Trying to override fixed parameter {setter.full_name}")
                graph_state.set(self.node_hash, name, given_args[setter.full_name])
            else:
                # Check if we need to sample this parameter's dependency node.
                if setter.dependency is not None and setter.dependency != self:
                    setter.dependency._sample_helper(graph_state, seen_nodes, given_args, rng_info)

                # Set the result from the correct source.
                if setter.source_type == ParameterSource.CONSTANT:
                    graph_state.set(self.node_hash, name, setter.value)
                elif setter.source_type == ParameterSource.MODEL_PARAMETER:
                    graph_state.set(
                        self.node_hash,
                        name,
                        graph_state[setter.dependency.node_hash][setter.value],
                    )
                elif setter.source_type == ParameterSource.FUNCTION_NODE:
                    graph_state.set(
                        self.node_hash,
                        name,
                        graph_state[setter.dependency.node_hash][setter.value],
                    )
                elif setter.source_type == ParameterSource.COMPUTE_OUTPUT:
                    # Computed parameters are set only after all the other (input) parameters.
                    any_compute = True
                else:
                    raise ValueError(f"Invalid ParameterSource type {setter.source_type}")

        # If this is a function node and the parameters depend on the result of its own computation
        # call the compute function to fill them in.
        if any_compute:
            self.compute(graph_state, given_args, rng_info)

    def sample_parameters(self, given_args=None, num_samples=1, rng_info=None):
        """Sample the model's underlying parameters if they are provided by a function
        or ParameterizedNode.

        Parameters
        ----------
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        num_samples : `int`
            A count of the number of samples to compute.
            Default: 1
        rng_info : `dict`, optional
            A dictionary of random number generator information for each node, such as
            the JAX keys or the numpy rngs.

        Returns
        -------
        graph_state : `GraphState`
            A dictionary of dictionaries mapping node->hash, variable_name to either a
            value or array of values. This data structure is modified in place to represent
            the model's state(s).

        Raises
        ------
        Raise a ``ValueError`` the sampling encounters a problem with the order of dependencies.
        """
        # If the graph structure has never been set, do that now.
        if self.node_pos is None:
            nodes = set()
            self.set_graph_positions(seen_nodes=nodes)

        # Resample the nodes. All information is stored in the returned results dictionary.
        seen_nodes = {}
        results = GraphState(num_samples)
        self._sample_helper(results, seen_nodes, given_args, rng_info)
        return results

    def get_all_node_info(self, field, seen_nodes=None):
        """Return a list of requested information for each node.

        Parameters
        ----------
        field : `str`
            The name of the attribute to extract from the node.
            Common examples are: "node_hash" and "node_string"
        seen_nodes : `set`
            A set of objects that have already been processed.
            Modified in place if provided.

        Returns
        -------
        result : `list`
            A list of values for each unique node in the graph.
        """
        # Check if the node might have incomplete information.
        if self.node_pos is None and (field == "node_pos" or field == "node_hash"):
            raise ValueError(
                f"Node {self.node_string} is missing position. You must call "
                f"set_graph_positions() before querying {field}."
            )

        # Check if we have already processed this node.
        if seen_nodes is None:
            seen_nodes = set()
        if self in seen_nodes:
            return []  # Nothing to do
        seen_nodes.add(self)

        # Get the information for this node and all its dependencies.
        result = [getattr(self, field)]
        for dep in self.direct_dependencies:
            result.extend(dep.get_all_node_info(field, seen_nodes))
        return result

    def build_pytree(self, graph_state, seen=None):
        """Build a JAX PyTree representation of the variables in this graph.

        Parameters
        ----------
        graph_state : `dict`
            A dictionary of dictionaries mapping node->hash, variable_name to value.
            This data structure is modified in place to represent the current state.
        seen : `set`
            A set of objects that have already been processed.
            Default : ``None``
        Returns
        -------
        values : `dict`
            The dictionary mapping the combination of the object identifier and
            model parameter name to its value.
        """
        # Check if the node might have incomplete information.
        if self.node_pos is None:
            raise ValueError(
                f"Node {self.node_string} is missing position. You must call "
                "set_graph_positions() before building a pytree."
            )

        # Skip nodes that we have already seen.
        if seen is None:
            seen = set()
        if self in seen:
            return {}
        seen.add(self)

        all_values = {}
        for name, setter_info in self.setters.items():
            if setter_info.dependency is not None:
                all_values.update(setter_info.dependency.build_pytree(graph_state, seen))
            elif setter_info.source_type == ParameterSource.CONSTANT and not setter_info.fixed:
                # Only the non-fixed, constants go into the PyTree.
                all_values[setter_info.full_name] = graph_state[self.node_hash][name]
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
            self.setters[name].set_as_compute_output(param_name=name)

    def _update_node_string(self, extra_tag=None):
        """Update the node's string."""
        if extra_tag is not None:
            super()._update_node_string(extra_tag)
        elif self.func is not None:
            super()._update_node_string(self.func.__name__)
        else:
            super()._update_node_string()

    def _build_inputs(self, graph_state, given_args=None, **kwargs):
        """Build the input arguments for the function.

        Parameters
        ----------
        graph_state : `GraphState`
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        **kwargs : `dict`, optional
            Additional function arguments.

        Returns
        -------
        args : `dict`
            A dictionary of input argument to value.
        """
        args = {}
        for key in self.arg_names:
            # Override with the given arg or kwarg in that order.
            if given_args is not None and self.setters[key].full_name in given_args:
                args[key] = given_args[self.setters[key].full_name]
            elif key in kwargs:
                args[key] = kwargs[key]
            else:
                args[key] = graph_state[self.node_hash][key]
        return args

    def _save_results(self, results, graph_state):
        """Save the results to the graph state.

        Parameters
        ----------
        results : iterable
            The function's results.
        graph_state : `GraphState`
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        """
        if len(self.outputs) == 1:
            graph_state.set(self.node_hash, self.outputs[0], results)
        else:
            if len(results) != len(self.outputs):
                raise ValueError(
                    f"Incorrect number of results returned by {self.func.__name__}. "
                    f"Expected {len(self.outputs)}, but got {results}."
                )
            for i in range(len(self.outputs)):
                graph_state.set(self.node_hash, self.outputs[i], results[i])

    def compute(self, graph_state, given_args=None, rng_info=None, **kwargs):
        """Execute the wrapped function.

        The input arguments are taken from the current graph_state and the outputs
        are written to graph_state.

        Parameters
        ----------
        graph_state : `GraphState`
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        rng_info : `dict`, optional
            A dictionary of random number generator information for each node, such as
            the JAX keys or the numpy rngs.
        **kwargs : `dict`, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.

        Raises
        ------
        ``ValueError`` is ``func`` attribute is ``None``.
        """
        if self.func is None:
            raise ValueError(
                "func parameter is None for a FunctionNode. You need to either "
                "set func or override compute()."
            )

        # Build a dictionary of arguments for the function, call the function, and save
        # the results in the graph state.
        args = self._build_inputs(graph_state, given_args, **kwargs)
        results = self.func(**args)
        self._save_results(results, graph_state)
        return results

    def resample_and_compute(self, given_args=None, rng_info=None):
        """A helper function for JAX gradients that runs the sampling then computation.

        Parameters
        ----------
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        rng_info : `dict`, optional
            A dictionary of random number generator information for each node, such as
            the JAX keys or the numpy rngs.
        """
        graph_state = self.sample_parameters(given_args, 1, rng_info)
        return self.compute(graph_state, given_args, rng_info)
