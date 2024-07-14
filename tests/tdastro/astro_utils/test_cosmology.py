import pytest
from astropy.cosmology import WMAP9
from tdastro.astro_utils.cosmology import redshift_to_distance


def test_redshift_to_distance():
    """Test that we can convert the redshift to a distance using a given cosmology."""
    # Use the example from:
    # https://docs.astropy.org/en/stable/api/astropy.cosmology.units.redshift_distance.html
    assert redshift_to_distance(1100, cosmology=WMAP9) == pytest.approx(14004.03157418 * 1e6)
