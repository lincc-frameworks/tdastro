"""The base PhysicalModel used for all sources."""

from tdastro.base_models import ParameterizedNode


class PhysicalModel(ParameterizedNode):
    """A physical model of a source of flux.

    Physical models can have fixed attributes (where you need to create a new model
    to change them) and settable attributes that can be passed functions or constants.
    They can also have special background pointers that link to another PhysicalModel
    producing flux. We can chain these to have a supernova in front of a star in front
    of a static background.

    Attributes
    ----------
    ra : `float`
        The object's right ascension (in degrees)
    dec : `float`
        The object's declination (in degrees)
    distance : `float`
        The object's distance (in pc).
    background : `PhysicalModel`
        A source of background flux such as a host galaxy.
    effects : `list`
        A list of effects to apply to an observations.
    """

    def __init__(self, ra=None, dec=None, distance=None, background=None, **kwargs):
        super().__init__(**kwargs)
        self.effects = []

        # Set RA, dec, and redshift from the parameters.
        self.add_parameter("ra", ra)
        self.add_parameter("dec", dec)
        self.add_parameter("distance", distance)

        # Background is an object not a sampled parameter
        self.background = background

    def __str__(self):
        """Return the string representation of the model."""
        return "PhysicalModel"

    def add_effect(self, effect, allow_dups=True, **kwargs):
        """Add a transformational effect to the PhysicalModel.
        Effects are applied in the order in which they are added.

        Parameters
        ----------
        effect : `EffectModel`
            The effect to apply.
        allow_dups : `bool`
            Allow multiple effects of the same type.
            Default = ``True``
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Raises
        ------
        Raises a ``AttributeError`` if the PhysicalModel does not have all of the
        required attributes.
        """
        # Check that we have not added this effect before.
        if not allow_dups:
            effect_type = type(effect)
            for prev in self.effects:
                if effect_type == type(prev):
                    raise ValueError("Added the effect type to a model {effect_type} more than once.")

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

        # Compute the flux density for both the current object and add in anything
        # behind it, such as a host galaxy.
        flux_density = self._evaluate(times, wavelengths, **kwargs)
        if self.background is not None:
            flux_density += self.background._evaluate(times, wavelengths, ra=self.ra, dec=self.dec, **kwargs)

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
        if self.background is not None:
            self.background.sample_parameters(include_effects, **kwargs)
        super().sample_parameters(**kwargs)

        if include_effects:
            for effect in self.effects:
                effect.sample_parameters(**kwargs)
