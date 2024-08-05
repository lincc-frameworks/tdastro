from astropy.cosmology import Planck18
from tdastro.sources.physical_model import PhysicalModel


def test_physical_model():
    """Test that we can create a PhysicalModel."""
    # Everything is specified.
    model1 = PhysicalModel(ra=1.0, dec=2.0, distance=3.0, redshift=0.0)
    assert model1["ra"] == 1.0
    assert model1["dec"] == 2.0
    assert model1["distance"] == 3.0
    assert model1["redshift"] == 0.0

    # Derive the distance from the redshift.
    model2 = PhysicalModel(ra=1.0, dec=2.0, redshift=1100.0, cosmology=Planck18)
    assert model2["ra"] == 1.0
    assert model2["dec"] == 2.0
    assert model2["redshift"] == 1100.0
    assert 13.0 * 1e12 < model2["distance"] < 16.0 * 1e12

    # Check that the RedshiftDistFunc node has the same computed value.
    # The syntax is a bit ugly because we are checking internal state.
    assert model2["distance"] == model2.setters["distance"].dependency["function_node_result"]

    # Neither distance nor redshift are specified.
    model3 = PhysicalModel(ra=1.0, dec=2.0)
    assert model3["redshift"] is None
    assert model3["distance"] is None

    # Redshift is specified but cosmology is not.
    model4 = PhysicalModel(ra=1.0, dec=2.0, redshift=1100.0)
    assert model4["redshift"] == 1100.0
    assert model4["distance"] is None
