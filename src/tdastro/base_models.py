"""The base models used to specify the TDAstro computation graph."""

import types
from enum import Enum


class ParameterSource(Enum):
    """ParameterSource specifies where a PhysicalModel should get the value
    for a given parameter: a constant value, a function, or from another
    parameterized model.
    """

    CONSTANT = 1
    FUNCTION = 2
    MODEL_ATTRIBUTE = 3
    MODEL_METHOD = 4


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
        A counter used to syncronize  sampling runs. Tracks how many times this
        model's parameters have been resampled.
    """

    def __init__(self, **kwargs):
        self.setters = {}
        self.sample_iteration = 0

    def __str__(self):
        """Return the string representation of the node."""
        return "ParameterizedNode"

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

        if value is not None:
            if isinstance(value, types.FunctionType):
                # Case 1: If we are getting from a static function, sample it.
                self.setters[name] = (ParameterSource.FUNCTION, value, required)
                setattr(self, name, value(**kwargs))
            elif isinstance(value, types.MethodType) and isinstance(value.__self__, ParameterizedNode):
                # Case 2: We are trying to use the method from a ParameterizedNode.
                # Note that this will (correctly) fail if we are adding a model method from the current
                # object that requires an unset attribute.
                self.setters[name] = (ParameterSource.MODEL_METHOD, value, required)
                setattr(self, name, value(**kwargs))
            elif isinstance(value, ParameterizedNode):
                # Case 3: We are trying to access an attribute from a ParameterizedNode.
                if not hasattr(value, name):
                    raise ValueError(f"Attribute {name} missing from parent.")
                self.setters[name] = (ParameterSource.MODEL_ATTRIBUTE, value, required)
                setattr(self, name, getattr(value, name))
            else:
                # Case 4: The value is constant.
                self.setters[name] = (ParameterSource.CONSTANT, value, required)
                setattr(self, name, value)
        elif not required:
            self.setters[name] = (ParameterSource.CONSTANT, None, required)
            setattr(self, name, None)
        else:
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
                # Check if we need to resample the parent (needs to be done before
                # we read its attribute).
                if setter.sample_iteration == self.sample_iteration:
                    setter.sample_parameters(max_depth - 1, **kwargs)
                sampled_value = getattr(setter, name)
            elif source_type == ParameterSource.MODEL_METHOD:
                # Check if we need to resample the parent (needs to be done before
                # we evaluate its method). Do not resample the current object.
                parent = setter.__self__
                if parent is not self and parent.sample_iteration == self.sample_iteration:
                    parent.sample_parameters(max_depth - 1, **kwargs)
                sampled_value = setter(**kwargs)
            else:
                raise ValueError(f"Unknown ParameterSource type {source_type} for {name}")
            setattr(self, name, sampled_value)

        # Increase the sampling iteration.
        self.sample_iteration += 1
