import numpy as np
import types

from tdastro.base_models import EffectModel, PhysicalModel

class WhiteNoise(EffectModel):
        def __init__(self, scale, **kwargs):
            super().__init__(**kwargs)
            self.scale = scale

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
            return np.random.normal(loc=flux_density, scale=self.scale)


class DistanceBasedWhiteNoise(EffectModel):
        def __init__(self, scale, dist_multiplier, **kwargs):
            super().__init__(**kwargs)
            self.scale = scale
            self.dist_multiplier = dist_multiplier

        def required_parameters(self):
            """Returns a list of the parameters of a PhysicalModel
            that this effect needs to access.

            Returns
            -------
            parameters : `list` of `str`
                A list of every required parameter the effect needs.
            """
            return ['distance']

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
            scale_value = self.scale + self.dist_multiplier * physical_model.distance
            return np.random.normal(loc=flux_density, scale=scale_value)
