class PhysicalModel:
    """A physical model of a source of flux.

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

        # Set RA, dec, and distance from the given value or, if it is None and
        # we have a host, from the host's value.
        self.ra = ra
        self.dec = dec
        self.distance = distance
        if ra is None and host is not None:
            self.ra = host.ra
        if dec is None and host is not None:
            self.dec = host.dec
        if distance is None and host is not None:
            self.distance = host.distance

        self.effects = []

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

    def _evaluate(self, times, wavelengths=None, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length N array of timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length N-array of flux densities.
        """
        raise NotImplementedError()

    def evaluate(self, times, wavelengths=None, **kwargs):
        """Draw observations for this object and apply the noise.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length N array of timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length N-array of flux densities.
        """
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
            A length N array of flux density values.
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
            A length N-array of flux densities after the effect is applied.
        """
        raise NotImplementedError()
