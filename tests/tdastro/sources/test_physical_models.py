import numpy as np
import pytest
from astropy.cosmology import Planck18
from tdastro.astro_utils.passbands import PassbandGroup
from tdastro.sources.physical_model import PhysicalModel
from tdastro.sources.static_source import StaticSource


def test_physical_model():
    """Test that we can create a PhysicalModel."""
    # Everything is specified.
    model1 = PhysicalModel(ra=1.0, dec=2.0, redshift=0.0)
    state = model1.sample_parameters()

    assert model1.get_param(state, "ra") == 1.0
    assert model1.get_param(state, "dec") == 2.0
    assert model1.get_param(state, "redshift") == 0.0
    assert model1.apply_redshift

    # None of the parameters are in the PyTree.
    pytree = model1.build_pytree(state)
    assert len(pytree["0:PhysicalModel"]) == 0

    # We can turn off the redshift computation.
    model1.set_apply_redshift(False)
    assert not model1.apply_redshift

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
    assert not model3.apply_redshift

    # Redshift is specified but cosmology is not.
    model4 = PhysicalModel(ra=1.0, dec=2.0, redshift=1100.0)
    state = model4.sample_parameters()
    assert model4.get_param(state, "redshift") == 1100.0
    assert model4.get_param(state, "distance") is None


def test_physical_model_get_all_node_info():
    """Test that we can query get_all_node_info from a PhysicalModel."""
    bg_model = PhysicalModel(ra=1.0, dec=2.0, distance=3.0, redshift=0.0, node_label="bg")
    source_model = PhysicalModel(
        ra=1.0,
        dec=2.0,
        distance=3.0,
        redshift=0.0,
        background=bg_model,
        node_label="source",
    )

    node_labels = source_model.get_all_node_info("node_label")
    assert len(node_labels) == 2
    assert "bg" in node_labels
    assert "source" in node_labels


def test_physical_model_build_np_rngs():
    """Test that we can build a dictionary of random number generators from a PhysicalModel."""
    bg_model = PhysicalModel(ra=1.0, dec=2.0, distance=3.0, redshift=0.0, node_label="bg")
    source_model = PhysicalModel(
        ra=1.0,
        dec=2.0,
        distance=3.0,
        redshift=0.0,
        background=bg_model,
        node_label="source",
    )
    np_rngs = source_model.build_np_rngs(base_seed=10)
    assert len(np_rngs) == 2


def test_physical_model_get_band_fluxes(passbands_dir):
    """Test that band fluxes are computed correctly."""
    # It should work fine for any positive Fnu.
    f_nu = np.random.lognormal()
    static_source = StaticSource(brightness=f_nu)
    state = static_source.sample_parameters()
    passbands = PassbandGroup(preset="LSST")

    times = np.arange(len(passbands), dtype=float)
    filters = np.array(sorted(passbands.passbands.keys()))

    # It should fail if no filters are provided.
    with pytest.raises(ValueError):
        _band_fluxes = static_source.get_band_fluxes(passbands, times=times, filters=None, state=state)
    # It should fail if single passband is provided, but with multiple filter names.
    with pytest.raises(ValueError):
        _band_fluxes = static_source.get_band_fluxes(passbands.passbands["LSST_r"], times, filters, state)

    band_fluxes = static_source.get_band_fluxes(passbands, times, filters, state)
    np.testing.assert_allclose(band_fluxes, f_nu, rtol=1e-10)
