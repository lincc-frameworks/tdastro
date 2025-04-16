import numpy as np
import pytest
from tdastro.effects.basic_effects import ConstantDimming
from tdastro.sources.basic_sources import StaticSource


def test_constant_dimming() -> None:
    """Test that we can sample and create a ConstantDimming."""
    values = np.full((5, 3), 100.0)

    # We can apply the noise.
    effect = ConstantDimming(flux_fraction=0.1)
    values = effect.apply(values, flux_fraction=0.1)
    assert np.all(values == 10.0)

    # We fail if we do not pass in the expected parameters.
    with pytest.raises(ValueError):
        _ = effect.apply(values)


def test_static_source_constant_dimming() -> None:
    """Test that we can sample and create a StaticSource object with constant dimming."""
    model = StaticSource(brightness=10.0, node_label="my_static_source")
    assert len(model.rest_frame_effects) == 0
    assert len(model.obs_frame_effects) == 0

    # We can add the white noise effect.  By default it is a rest frame effect.
    effect = ConstantDimming(flux_fraction=0.1)
    model.add_effect(effect)
    assert len(model.rest_frame_effects) == 1
    assert len(model.obs_frame_effects) == 0

    state = model.sample_parameters()
    times = np.array([1, 2, 3, 4, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])
    values = model.evaluate(times, wavelengths, state)
    assert values.shape == (6, 3)
    assert np.all(values == 1.0)

    # We can add the white noise effect as a observer frame effect instead.
    model2 = StaticSource(brightness=10.0, node_label="my_static_source")
    effect2 = ConstantDimming(flux_fraction=0.5, rest_frame=False)
    model2.add_effect(effect2)
    assert len(model2.rest_frame_effects) == 0
    assert len(model2.obs_frame_effects) == 1

    state2 = model2.sample_parameters()
    values2 = model2.evaluate(times, wavelengths, state2)
    assert values2.shape == (6, 3)
    assert np.all(values2 == 5.0)
