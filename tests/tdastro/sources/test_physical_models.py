from astropy.cosmology import Planck18
from tdastro.sources.physical_model import PhysicalModel


def test_physical_model():
    """Test that we can create a PhysicalModel."""
    # Everything is specified.
    model1 = PhysicalModel(ra=1.0, dec=2.0, distance=3.0, redshift=0.0)
    state = model1.sample_parameters()

    assert model1.get_param(state, "ra") == 1.0
    assert model1.get_param(state, "dec") == 2.0
    assert model1.get_param(state, "distance") == 3.0
    assert model1.get_param(state, "redshift") == 0.0

    # Derive the distance from the redshift.
    model2 = PhysicalModel(ra=1.0, dec=2.0, redshift=1100.0, cosmology=Planck18)
    state = model2.sample_parameters()
    assert model2.get_param(state, "ra") == 1.0
    assert model2.get_param(state, "dec") == 2.0
    assert model2.get_param(state, "redshift") == 1100.0
    assert 13.0 * 1e12 < model2.get_param(state, "distance") < 16.0 * 1e12

    # Check that the RedshiftDistFunc node has the same computed value.
    # The syntax is a bit ugly because we are checking internal state.
    model2_val = model2.get_param(state, "distance")
    func_val = model2.setters["distance"].dependency.get_param(state, "function_node_result")
    assert model2_val == func_val

    # Neither distance nor redshift are specified.
    model3 = PhysicalModel(ra=1.0, dec=2.0)
    state = model3.sample_parameters()
    assert model3.get_param(state, "redshift") is None
    assert model3.get_param(state, "distance") is None

    # Redshift is specified but cosmology is not.
    model4 = PhysicalModel(ra=1.0, dec=2.0, redshift=1100.0)
    state = model4.sample_parameters()
    assert model4.get_param(state, "redshift") == 1100.0
    assert model4.get_param(state, "distance") is None
