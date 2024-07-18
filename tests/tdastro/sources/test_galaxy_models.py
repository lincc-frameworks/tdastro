import random

import numpy as np
from tdastro.sources.galaxy_models import GaussianGalaxy
from tdastro.sources.static_source import StaticSource
from tdastro.util_nodes.np_random import NumpyRandomFunc


def _sample_ra(**kwargs):
    """Return a random value between 0 and 360.

    Parameters
    ----------
    **kwargs : `dict`, optional
        Absorbs additional parameters
    """
    return 360.0 * random.random()


def _sample_dec(**kwargs):
    """Return a random value between -90 and 90.

    Parameters
    ----------
    **kwargs : `dict`, optional
        Absorbs additional parameters
    """
    return 180.0 * random.random() - 90.0


def test_gaussian_galaxy() -> None:
    """Test that we can sample and create a StaticSource object."""
    random.seed(1001)

    host = GaussianGalaxy(ra=_sample_ra, dec=_sample_dec, brightness=10.0, radius=1.0 / 3600.0)
    host_ra = host.ra
    host_dec = host.dec

    # We define the position of the source using Gaussian noise from the center
    # of the host galaxy.
    ra_func = NumpyRandomFunc(
        "normal",
        loc=(host, "ra"),
        scale=(host, "galaxy_radius_std"),
        node_identifier="ra_host_noise",
        graph_base_seed=1001,
    )
    dec_func = NumpyRandomFunc(
        "normal",
        loc=(host, "dec"),
        scale=(host, "galaxy_radius_std"),
        node_identifier="dec_host_noise",
        graph_base_seed=1001,
    )
    source = StaticSource(ra=ra_func, dec=dec_func, background=host, brightness=100.0)

    # Both RA and dec should be "close" to (but not exactly at) the center of the galaxy.
    source_ra_offset = source.ra - host_ra
    assert 0.0 < np.abs(source_ra_offset) < 100.0 / 3600.0

    source_dec_offset = source.dec - host_dec
    assert 0.0 < np.abs(source_dec_offset) < 100.0 / 3600.0

    times = np.array([1, 2, 3, 4, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])

    # All the measured fluxes should have some contribution from the background object.
    values = source.evaluate(times, wavelengths)
    assert values.shape == (6, 3)
    assert np.all(values > 100.0)
    assert np.all(values <= 110.0)

    # Check that if we resample the source it will propagate and correctly resample the host.
    # the host's (RA, dec) should change and the source's should still be close.
    source.sample_parameters()
    assert host_ra != host.ra
    assert host_dec != host.dec

    source_ra_offset2 = source.ra - host.ra
    assert source_ra_offset != source_ra_offset2
    assert 0.0 < np.abs(source_ra_offset2) < 100.0 / 3600.0

    source_dec_offset2 = source.dec - host.dec
    assert source_dec_offset != source_dec_offset2
    assert 0.0 < np.abs(source_ra_offset2) < 100.0 / 3600.0
