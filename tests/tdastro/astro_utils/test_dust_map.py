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
        The DustMap's ebv value at all points in the sky.
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


class TestExtinction:
    """An extinction function that computes the scaling factor
    as a constant times the ebv.

    Attributes
    ----------
    scale : `float`
        The extinction function's multiplicative scaling.
    """

    def __init__(self, scale):
        self.scale = scale

    def extinguish(self, wavelengths, Ebv=1.0):
        """The extinguish function

        Parameters
        ----------
        wavelengths : numpy.ndarray
            The array of wavelengths
        Ebv : float
            The Ebv to use in the scaling.
        """
        return Ebv * self.scale


def test_load_extinction_model():
    """Load an extinction model by string."""
    g23_model = DustExtinctionEffect.load_extinction_model("G23", Rv=3.1)
    assert g23_model is not None
    assert hasattr(g23_model, "extinguish")

    # Load through the DustExtinctionEffect constructor.
    const_map = ConstantDustMap(0.5)
    dust_effect = DustExtinctionEffect(const_map, "CCM89", Rv=3.1)
    assert dust_effect.extinction_model is not None
    assert hasattr(dust_effect.extinction_model, "extinguish")


def test_constant_dust_extinction():
    """Test that we can create and sample a DustExtinctionEffect object."""
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    wavelengths = np.array([7000.0, 5200.0, 4800.0])  # Red, green, blue

    # Create a model without any dust.
    model_clean = StaticSource(brightness=100.0, ra=0.0, dec=40.0)
    state = model_clean.sample_parameters()
    fluxes_clean = model_clean.evaluate(times, wavelengths, state)
    assert fluxes_clean.shape == (5, 3)
    assert np.all(fluxes_clean == 100.0)

    # Create DustExtinctionEffect effects with constant ebvs and constant
    # multiplicative extinction functions.
    dust_effect = DustExtinctionEffect(ConstantDustMap(0.5), TestExtinction(0.1))
    fluxes = dust_effect.apply(fluxes_clean, wavelengths, ra=0.0, dec=40.0, dist=100.0)
    assert fluxes.shape == (5, 3)
    assert np.all(fluxes == 5.0)

    dust_effect = DustExtinctionEffect(ConstantDustMap(0.5), TestExtinction(0.3))
    fluxes = dust_effect.apply(fluxes_clean, wavelengths, ra=0.0, dec=40.0, dist=100.0)
    assert fluxes.shape == (5, 3)
    assert np.all(fluxes == 15.0)

    # Use a manual ebv, which overrides the dustmap.
    dust_effect = DustExtinctionEffect(ConstantDustMap(0.5), TestExtinction(0.1))
    fluxes = dust_effect.apply(fluxes_clean, wavelengths, ebv=1.0, ra=0.0, dec=40.0, dist=100.0)
    assert fluxes.shape == (5, 3)
    assert np.all(fluxes == 10.0)

    # Create a model with ccm89 extinction at r_v = 3.1.
    dust_effect = DustExtinctionEffect(ConstantDustMap(0.5), CCM89(Rv=3.1))
    fluxes_ccm98 = dust_effect.apply(fluxes_clean, wavelengths, ra=0.0, dec=40.0, dist=1.0)
    assert fluxes_ccm98.shape == (5, 3)
    assert np.all(fluxes_ccm98 < 100.0)
