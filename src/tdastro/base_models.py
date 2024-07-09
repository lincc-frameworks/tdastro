import types
from enum import Enum

import numpy as np


class ParameterSource(Enum):
    """ParameterSource specifies where a PhysicalModel should get the value
    for a given parameter: a constant value, a function, or from another
    parameterized model.
    """

    CONSTANT = 1
    FUNCTION = 2
    MODEL_ATTRIBUTE = 3
    MODEL_METHOD = 4


class ParameterizedModel:
    """Any model that uses parameters that can be set by constants,
    functions, or other parameterized models. ParameterizedModels can
    include physical objects or statistical distributions.

    Attributes
    ----------
    setters : `list` of `tuple`
        A dictionary to information about the setters for the parameters in the form:
        (name, ParameterSource, setter information). The attributes are stored in the
        order in which they need to be set.
    sample_iteration : `int`
        A counter used to syncronize  sampling runs. Tracks how many times this
        model's parameters have been resampled.
    """

    def __init__(self, **kwargs):
        """Create a ParameterizedModel object.

        Parameters
        ----------
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        self.setters = []
        self.sample_iteration = 0

    def __str__(self):
        """Return the string representation of the model."""
        return "ParameterizedModel"

    def add_parameter(self, name, value=None, required=False, **kwargs):
        """Add a single parameter to the ParameterizedModel. Checks multiple sources
        in the following order: 1. Manually specified ``value``, 2. An entry in ``kwargs``,
        or 3. ``None``.
        Sets an initial value for the attribute based on the given information.
        The attributes are stored in the order in which they are added.

        Parameters
        ----------
        name : `str`
            The parameter name to add.
        value : any, optional
            The information to use to set the parameter. Can be a constant,
            function, ParameterizedModel, or self.
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
        if hasattr(self, name) and getattr(self, name) is not None:
            raise KeyError(f"Duplicate parameter set: {KeyError}")

        if value is None and name in kwargs:
            # The value wasn't set, but the name is in kwargs.
            value = kwargs[name]
        if value is not None:
            if isinstance(value, types.FunctionType):
                # Case 1: If we are getting from a static function, sample it.
                self.setters.append((name, ParameterSource.FUNCTION, value))
                setattr(self, name, value(**kwargs))
            elif isinstance(value, types.MethodType) and isinstance(value.__self__, ParameterizedModel):
                # Case 2: We are trying to use the method from a ParameterizedModel.
                # Note that this will (correctly) fail if we are adding a model method from the current
                # object that requires an unset attribute.
                self.setters.append((name, ParameterSource.MODEL_METHOD, value))
                setattr(self, name, value(**kwargs))
            elif isinstance(value, ParameterizedModel):
                # Case 3: We are trying to access an attribute from a parameterized model.
                if not hasattr(value, name):
                    raise ValueError(f"Attribute {name} missing from parent.")
                self.setters.append((name, ParameterSource.MODEL_ATTRIBUTE, value))
                setattr(self, name, getattr(value, name))
            else:
                # Case 4: The value is constant.
                self.setters.append((name, ParameterSource.CONSTANT, value))
                setattr(self, name, value)
        elif not required:
            self.setters.append((name, ParameterSource.CONSTANT, None))
            setattr(self, name, None)
        else:
            raise ValueError(f"Missing required parameter {name}")

    def sample_parameters(self, max_depth=50, **kwargs):
        """Sample the model's underlying parameters if they are provided by a function
        or ParameterizedModel.

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
        for name, source_type, setter in self.setters:
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


class PhysicalModel(ParameterizedModel):
    """A physical model of a source of flux. Physical models can have fixed attributes
    (where you need to create a new model to change them) and settable attributes that
    can be passed functions or constants.

    Attributes
    ----------
    ra : `float`
        The object's right ascension (in degrees)
    dec : `float`
        The object's declination (in degrees)
    distance : `float`
        The object's distance (in pc)
    effects : `list`
        A list of effects to apply to an observations.
    """

    def __init__(self, ra=None, dec=None, distance=None, **kwargs):
        """Create a PhysicalModel object.

        Parameters
        ----------
        ra : `float`, `function`, or `ParameterizedModel`, optional
            The object's right ascension (in degrees)
        dec : `float`, `function`, or `ParameterizedModel`, optional
            The object's declination (in degrees)
        distance : `float`, `function`, or `ParameterizedModel`, optional
            The object's distance (in pc)
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.effects = []

        # Set RA, dec, and distance from the parameters.
        self.add_parameter("ra", ra)
        self.add_parameter("dec", dec)
        self.add_parameter("distance", distance)

    def __str__(self):
        """Return the string representation of the model."""
        return "PhysicalModel"

    def add_effect(self, effect, **kwargs):
        """Add a transformational effect to the PhysicalModel.
        Effects are applied in the order in which they are added.

        Parameters
        ----------
        effect : `EffectModel`
            The effect to apply.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Raises
        ------
        Raises a ``AttributeError`` if the PhysicalModel does not have all of the
        required attributes.
        """
        # Try applying this effect to a test flux to confirm that the effect
        # has all the information that it needs. This will fail if the PhysicalModel
        # is missing a required attribute.
        test_flux = np.full((1, 1), 1.0)
        test_wavelengths = np.full((1, 1), 1000.0)
        _ = effect.apply(test_flux, test_wavelengths, self, **kwargs)

        self.effects.append(effect)

    def _evaluate(self, times, wavelengths, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values.
        """
        raise NotImplementedError()

    def evaluate(self, times, wavelengths, resample_parameters=False, **kwargs):
        """Draw observations for this object and apply the noise.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        resample_parameters : `bool`
            Treat this evaluation as a completely new object, resampling the
            parameters from the original provided functions.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values.
        """
        if resample_parameters:
            self.sample_parameters(kwargs)

        flux_density = self._evaluate(times, wavelengths, **kwargs)
        for effect in self.effects:
            flux_density = effect.apply(flux_density, wavelengths, self, **kwargs)
        return flux_density

    def sample_parameters(self, include_effects=True, **kwargs):
        """Sample the model's underlying parameters if they are provided by a function
        or ParameterizedModel.

        Parameters
        ----------
        include_effects : `bool`
            Resample the parameters for the effects models.
        **kwargs : `dict`, optional
            All the keyword arguments, including the values needed to sample
            parameters.
        """
        super().sample_parameters(**kwargs)
        if include_effects:
            for effect in self.effects:
                effect.sample_parameters(**kwargs)


class EffectModel(ParameterizedModel):
    """A physical or systematic effect to apply to an observation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        """Return the string representation of the model."""
        return "EffectModel"

    def apply(self, flux_density, wavelengths=None, physical_model=None, **kwargs):
        """Apply the effect to observations (flux_density values)

        Parameters
        ----------
        flux_density : `numpy.ndarray`
            A length T X N matrix of flux density values.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        physical_model : `PhysicalModel`
            A PhysicalModel from which the effect may query parameters
            such as redshift, position, or distance.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of flux densities after the effect is applied.
        """
        raise NotImplementedError()
