from astropy.cosmology import WMAP9, Planck18
from tdastro.astro_utils.cosmology import redshift_to_distance


def test_redshift_to_distance():
    """Test that we can convert the redshift to a distance using a given cosmology."""
    wmap9_val = redshift_to_distance(1100, cosmology=WMAP9)
    planck18_val = redshift_to_distance(1100, cosmology=Planck18)

    assert abs(planck18_val - wmap9_val) > 1000.0
    assert 13.0 * 1e12 < wmap9_val < 16.0 * 1e12
    assert 13.0 * 1e12 < planck18_val < 16.0 * 1e12
