class PhysicalModel:
    """A physical model of a source of flux.

    Attributes
    ----------
    host : `PhysicalModel`
        A physical model of the current source's host.
    effects : `list`
        A list of effects to apply to an observations.
    """

    def __init__(self, host=None, **kwargs):
        """Create a PhysicalModel object.

        Parameters
        ----------
        host : `PhysicalModel`, optional
            A physical model of the current source's host.
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        self.host = host
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

    def _evaluate(self, times, bands=None, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            An array of timestamps.
        bands : `numpy.ndarray`, optional
            An array of bands.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            The results.
        """
        raise NotImplementedError()

    def evaluate(self, times, bands=None, **kwargs):
        """Draw observations for this object and apply the noise.

        Parameters
        ----------
        times : `numpy.ndarray`
            An array of timestamps.
        bands : `numpy.ndarray`, optional
            An array of bands.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            The results.
        """
        flux_density = self._evaluate(times, bands, **kwargs)
        for effect in self.effects:
            flux_density = effect.apply(flux_density, bands, self, **kwargs)
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

    def apply(self, flux_density, bands=None, physical_model=None, **kwargs):
        """Apply the effect to observations (flux_density values)

        Parameters
        ----------
        flux_density : `numpy.ndarray`
            An array of flux density values.
        bands : `numpy.ndarray`, optional
            An array of bands.
        physical_model : `PhysicalModel`
            A PhysicalModel from which the effect may query parameters
            such as redshift, position, or distance.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            The results.
        """
        raise NotImplementedError()
