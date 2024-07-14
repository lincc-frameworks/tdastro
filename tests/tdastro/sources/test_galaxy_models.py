import random

import numpy as np
from tdastro.sources.galaxy_models import GaussianGalaxy
from tdastro.sources.static_source import StaticSource


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

    source = StaticSource(ra=host.sample_ra, dec=host.sample_dec, background=host, brightness=100.0)

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
    print(f"Host sample = {host.sample_iteration}")
    print(f"Source sample = {source.sample_iteration}")
    source.sample_parameters()
    print(f"Host sample = {host.sample_iteration}")
    print(f"Source sample = {source.sample_iteration}")
    assert host_ra != host.ra
    assert host_dec != host.dec

    source_ra_offset2 = source.ra - host.ra
    assert source_ra_offset != source_ra_offset2
    assert 0.0 < np.abs(source_ra_offset2) < 100.0 / 3600.0

    source_dec_offset2 = source.dec - host.dec
    assert source_dec_offset != source_dec_offset2
    assert 0.0 < np.abs(source_ra_offset2) < 100.0 / 3600.0
