import numpy as np
import pytest
from tdastro.effects.extinction import ExtinctionEffect
from tdastro.math_nodes.given_sampler import GivenValueList
from tdastro.sources.basic_sources import StaticSource


def test_list_extinction_models():
    """List the available extinction models."""
    model_names = ExtinctionEffect.list_extinction_models()
    assert len(model_names) > 10
    assert "G23" in model_names
    assert "CCM89" in model_names


def test_load_extinction_model():
    """Load an extinction model by string."""
    g23_model = ExtinctionEffect.load_extinction_model("G23", Rv=3.1)
    assert g23_model is not None
    assert hasattr(g23_model, "extinguish")

    # We fail if we try to load a model that does not exist.
    with pytest.raises(KeyError):
        ExtinctionEffect.load_extinction_model("InvalidModel")

    # We can manually load the g23_model into an ExtinctionEffect node.
    dust_effect = ExtinctionEffect(g23_model, ebv=0.1)

    # We can apply the extinction effect to a set of fluxes.
    fluxes = np.full((10, 3), 1.0)
    wavelengths = np.array([7000.0, 5200.0, 4800.0])
    new_fluxes = dust_effect.apply(fluxes, wavelengths=wavelengths, ebv=0.1)
    assert new_fluxes.shape == (10, 3)
    assert np.all(new_fluxes < fluxes)

    # We fail if we are missing a required parameter.
    with pytest.raises(ValueError):
        _ = dust_effect.apply(fluxes, wavelengths=wavelengths)
    with pytest.raises(ValueError):
        _ = dust_effect.apply(fluxes, ebv=0.1)


def test_constant_dust_extinction():
    """Test that we can create and sample a ExtinctionEffect object."""
    # Use given ebv values. Usually these would be computed from a dustmap,
    # based on (RA, dec).
    ebv_node = GivenValueList([0.1, 0.2, 0.3, 0.4, 0.5])
    dust_effect = ExtinctionEffect("CCM89", ebv=ebv_node, Rv=3.1)
    assert dust_effect.extinction_model is not None
    assert hasattr(dust_effect.extinction_model, "extinguish")

    model = StaticSource(
        brightness=100.0,
        ra=0.0,
        dec=40.0,
        redshift=0.0,
        effects=[dust_effect],
    )

    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    wavelengths = np.array([7000.0, 5200.0, 4800.0])  # Red, green, blue
    states = model.sample_parameters(num_samples=3)
    fluxes = model.evaluate(times, wavelengths, states)

    assert fluxes.shape == (3, 5, 3)
    assert np.all(fluxes < 100.0)
    assert np.all(fluxes[0, :, :] > fluxes[1, :, :])
    assert np.all(fluxes[1, :, :] > fluxes[2, :, :])
