import pytest
from astropy.cosmology import WMAP9
from tdastro.sources.physical_model import PhysicalModel


def test_physical_model():
    """Test that we can create a physical model."""
    # Everything is specified.
    model1 = PhysicalModel(ra=1.0, dec=2.0, distance=3.0, redshift=0.0)
    assert model1.ra == 1.0
    assert model1.dec == 2.0
    assert model1.distance == 3.0
    assert model1.redshift == 0.0

    # Derive the distance from the redshift using the example from:
    # https://docs.astropy.org/en/stable/api/astropy.cosmology.units.redshift_distance.html
    model2 = PhysicalModel(ra=1.0, dec=2.0, redshift=1100.0, cosmology=WMAP9)
    assert model2.ra == 1.0
    assert model2.dec == 2.0
    assert model2.redshift == 1100.0
    assert model2.distance == pytest.approx(14004.03157418 * 1e6)

    # Neither distance nor redshift are specified.
    model3 = PhysicalModel(ra=1.0, dec=2.0)
    assert model3.redshift is None
    assert model3.distance is None

    # Redshift is specified but cosmology is not.
    model4 = PhysicalModel(ra=1.0, dec=2.0, redshift=1100.0)
    assert model4.redshift == 1100.0
    assert model4.distance is None
