import abc

class PhysicalModel(abc.ABC):
       def __init__(self, ra=None, dec=None, distance=None, **kwargs):
            self.ra = ra
            self.dec = dec
            self.distance = distance
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
                if (getattr(self, parameter) is None):
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

            Returns
            -------
            flux_density : `numpy.ndarray`
                The results.
            """
            flux_density = self._evaluate(times, bands, **kwargs)
            for effect in self.effects:
                flux_density = effect.apply(flux_density, bands, self, **kwargs)
            return flux_density


class EffectModel(abc.ABC):
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

            Returns
            -------
            flux_density : `numpy.ndarray`
                The results.
            """
            raise NotImplementedError()