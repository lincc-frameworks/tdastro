import numpy as np

from dust_extinction.parameter_averages import CCM89
from dustmaps.map_base import DustMap

from tdastro.astro_utils.dust_map import DustExtinctionEffect
from tdastro.sources.static_source import StaticSource


class ConstantDustMap(DustMap):
    """A DustMap with a constant value. Used for testing.
    
    Attributes
    ----------
    ebv : `float`
        The DustMap's ebv value at all 
    """
    def __init__(self, ebv):
        self.ebv = ebv

    def query(self, coords):
        """Returns reddening at the requested coordinates.
        
        Parameters
        ----------
        coords : `astropy.coordinates.SkyCoord`
            The coordinates of the query or queries.

        Returns
        -------
        Reddening : `float` or `numpy.ndarray`
            The result of the query.
        """
        if coords.isscalar:
            return self.ebv
        return np.full((len(coords)), self.ebv)


def test_constant_dust_extinction():
    """Test that we can create and sample a DustExtinctionEffect object."""
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    wavelengths = np.array([800.0, 900.0, 1000.0, 1000.0, 900.0])

    # Create a model without any dust.
    model_clean = StaticSource(brightness=100.0, ra=0.0, dec=40.0)
    state = model_clean.sample_parameters()
    fluxes_clean = model_clean.evaluate(times, wavelengths, state)
    assert len(fluxes_clean) == 5
    assert np.all(fluxes_clean == 100.0)

    # Create a model with ccm89 extinction at r_v = 3.1.
    dust_effect = DustExtinctionEffect(ConstantDustMap(0.5), CCM89(Rv=3.1))
    fluxes_ccm98 = dust_effect.apply(fluxes_clean, wavelengths, 0.0, 40.0, 1.0)
    assert len(fluxes_ccm98) == 5
    assert np.all(fluxes_ccm98 < 100.0)
