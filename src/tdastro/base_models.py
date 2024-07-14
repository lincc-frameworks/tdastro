"""The base models used to specify the TDAstro computation graph."""

import types
from enum import Enum


class ParameterSource(Enum):
    """ParameterSource specifies where a ParameterizedNode should get the value
    for a given parameter: a constant value or from another ParameterizedNode.
    """

    CONSTANT = 1
    MODEL_ATTRIBUTE = 2
    MODEL_METHOD = 3
    FUNCTION = 4


class ParameterizedNode:
    """Any model that uses parameters that can be set by constants,
    functions, or other parameterized nodes.

    Attributes
    ----------
    setters : `dict` of `tuple`
        A dictionary to information about the setters for the parameters in the form:
        (ParameterSource, setter information, required). The attributes are
        stored in the order in which they need to be set.
    sample_iteration : `int`
        A counter used to syncronize sampling runs. Tracks how many times this
        model's parameters have been resampled.
    """

    def __init__(self, **kwargs):
        self.setters = {}
        self.sample_iteration = 0

    def __str__(self):
        """Return the string representation of the model."""
        return "ParameterizedNode"

    def check_resample(self, child):
        """Check if we need to resample the current node based
        on the state of a child trying to access its attributes.

        Parameters
        ----------
        child : `ParameterizedNode`
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
        if child == self:
            return False
        if child.sample_iteration == self.sample_iteration:
            return False
        if child.sample_iteration != self.sample_iteration + 1:
            raise ValueError(
                f"Node {str(child)} at iteration {child.sample_iteration} accessing"
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
                self.setters[name] = (ParameterSource.MODEL_METHOD, value, required)
            else:
                # Case 1c: This is a general callable method from another object.
                # We treat it as static (we don't resample the other object).
                self.setters[name] = (ParameterSource.FUNCTION, value, required)
            setattr(self, name, value(**kwargs))
        elif isinstance(value, ParameterizedNode):
            # Case 2: We are trying to access a parameter of another ParameterizedNode.
            if not hasattr(value, name):
                raise ValueError(f"Attribute {name} missing from parent.")
            self.setters[name] = (ParameterSource.MODEL_ATTRIBUTE, value, required)
            setattr(self, name, getattr(value, name))
        else:
            # Case 3: The value is constant (including None).
            self.setters[name] = (ParameterSource.CONSTANT, value, required)
            setattr(self, name, value)

        if required and getattr(self, name) is None:
            raise ValueError(f"Missing required parameter {name}")

    def add_parameter(self, name, value=None, required=False, **kwargs):
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
        **kwargs : `dict`, optional
           All other keyword arguments, possibly including the parameter setters.

        Raises
        ------
        Raise a ``KeyError`` if there is a parameter collision or the parameter
        cannot be found.
        Raise a ``ValueError`` if the parameter is required, but set to None.
        """
        # Check for parameter collision.
        if hasattr(self, name) and getattr(self, name) is not None:
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


class FunctionNode(ParameterizedNode):
    """A class to wrap functions and their argument settings.

    Attributes
    ----------
    func : `function` or `method`
        The function to call during an evaluation.
    args_names : `list`
        A list of argument names to pass to the function.

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

    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.arg_names = []

        # Add all of the parameters from default_args or the kwargs.
        for key, value in kwargs.items():
            self.arg_names.append(key)
            self.add_parameter(key, value)

    def __str__(self):
        """Return the string representation of the function."""
        return f"FunctionNode({self.func.name})"

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
