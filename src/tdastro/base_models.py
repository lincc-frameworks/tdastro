import types
from enum import Enum


class ParameterSource(Enum):
    """ParameterSource specifies where a PhysicalModel should get the value
    for a given parameter: a constant value, a function, or from the host.
    """

    # TODO support something like TensorFlow probability to provide a DAG
    # of sampling dependencies.
    CONSTANT = 1
    FUNCTION = 2
    HOST = 3


class PhysicalModel:
    """A physical model of a source of flux. Physical models can have fixed attributes
    (where you need to create a new model to change them) and settable attributes that
    can be passed functions or constants.

    Attributes
    ----------
    host : `PhysicalModel`
        A physical model of the current source's host.
    ra : `float`
        The object's right ascension (in degrees)
    dec : `float`
        The object's declination (in degrees)
    distance : `float`
        The object's distance (in au)
    setters : `dict` or `tuple`
        A dictionary to information about the setters for the parameters in the form:
        (ParameterSource, value).
    effects : `list`
        A list of effects to apply to an observations.
    """

    def __init__(self, host=None, ra=None, dec=None, distance=None, **kwargs):
        """Create a PhysicalModel object.

        Parameters
        ----------
        host : `PhysicalModel`, optional
            A physical model of the current source's host.
        ra : `float`, optional
            The object's right ascension (in degrees)
        dec : `float`, optional
            The object's declination (in degrees)
        distance : `float`, optional
            The object's distance (in au)
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        self.host = host
        self.effects = []
        self.setters = {}

        # Set RA, dec, and distance from the parameters.
        self.add_parameter("ra", ra)
        self.add_parameter("dec", dec)
        self.add_parameter("distance", distance)

    def add_parameter(self, name, value=None, required=False, **kwargs):
        """Add a single parameter to the PhysicalModel. Checks multiple sources
        in the following order:
            1. Manually specified ``value``
            2. An entry in ``kwargs``
            3. An entry in the object's host.
            4. ``None``
        Sets an initial value for the attribute based on the given information.

        Parameters
        ----------
        name : `str`
            The parameter name to add.
        value : any, optional
            The information to use to set the parameter. Can be a constant
            or a function.
        required : `bool`
            Fail if the parameter is set to ``None``.
        **kwargs : `dict`, optional
           All other keyword arguments, possibly including the parameter setters.

        Raises
        ------
        Raise a ``KeyError`` if there is a parameter collision.
        Raise a ``ValueError`` if the parameter is required, but set to None.
        """
        if hasattr(self, name) and getattr(self, name) is not None:
            raise KeyError(f"Duplicate parameter set: {KeyError}")

        if value is None and name in kwargs:
            # The value wasn't set, but the name is in kwargs.
            value = kwargs[name]
        if value is not None:
            if isinstance(value, types.FunctionType):
                self.setters[name] = (ParameterSource.FUNCTION, value)
                setattr(self, name, value(**kwargs))
            else:
                self.setters[name] = (ParameterSource.CONSTANT, value)
                setattr(self, name, value)
        elif self.host is not None and hasattr(self.host, name):
            self.setters[name] = (ParameterSource.HOST, name)
            setattr(self, name, getattr(self.host, name))
        elif not required:
            self.setters[name] = (ParameterSource.CONSTANT, None)
            setattr(self, name, None)
        else:
            raise ValueError(f"Missing required parameter {name}")

    def sample_parameters(self, **kwargs):
        """Sample the model's underlying parameters if they are provided by a function.

        Parameters
        ----------
        **kwargs : `dict`, optional
           All the keyword arguments, including the values needed to sample parameters.
        """
        if self.host is not None:
            self.host.sample_parameters(**kwargs)

        # Run through each parameter and sample it based on the given recipe.
        for param, value in self.setters.items():
            sampled_value = None
            if value[0] == ParameterSource.CONSTANT:
                sampled_value = value[1]
            elif value[0] == ParameterSource.FUNCTION:
                sampled_value = value[1](**kwargs)
            elif value[0] == ParameterSource.HOST:
                sampled_value = self.host.getattr(param)
            else:
                raise ValueError(f"Unknown ParameterSource type {value[0]}")
            setattr(self, param, sampled_value)

    def add_effect(self, effect):
        """Add a transformational effect to the PhysicalModel.
        Effects are applied in the order in which they are added.

        Parameters
        ----------
        effect : `EffectModel`
            The effect to apply.

        Raises
        ------
        Raises a ``AttributeError`` if the PhysicalModel does not have all of the
        required attributes.
        """
        required: list = effect.required_parameters()
        for parameter in required:
            # Raise an AttributeError if the parameter is missing or set to None.
            if getattr(self, parameter) is None:
                raise AttributeError(f"Parameter {parameter} unset for model {type(self).__name__}")

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

        for effect in self.effects:
            if hasattr(effect, "pre_effect"):  # pre-effects == adjustments done to times and/or wavelengths
                times, wavelengths = effect.pre_effect(times, wavelengths, **kwargs)

        flux_density = self._evaluate(times, wavelengths, **kwargs)

        for effect in self.effects:
            flux_density = effect.apply(flux_density, wavelengths, self, **kwargs)

        return flux_density


class EffectModel:
    """A physical or systematic effect to apply to an observation."""

    def __init__(self, **kwargs):
        pass

    def required_parameters(self):
        """Returns a list of the parameters of a PhysicalModel
        that this effect needs to access.

        Returns
        -------
        parameters : `list` of `str`
            A list of every required parameter the effect needs.
        """
        return []

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
