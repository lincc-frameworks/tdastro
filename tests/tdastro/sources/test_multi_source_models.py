import numpy as np
import pytest
from tdastro.effects.basic_effects import ConstantDimming
from tdastro.math_nodes.np_random import NumpyRandomFunc
from tdastro.sources.basic_sources import StaticSource, StepSource
from tdastro.sources.multi_source_model import AdditiveMultiSourceModel, RandomMultiSourceModel


def test_additive_multi_source_node() -> None:
    """Test that we can create and evaluate a AdditiveMultiSourceModel."""
    source1 = StaticSource(brightness=10.0, node_label="my_static_source")
    source2 = StepSource(brightness=15.0, t0=1.0, t1=2.0, node_label="my_step_source")
    model = AdditiveMultiSourceModel([source1, source2], node_label="my_multi_source")

    state = model.sample_parameters()
    assert state["my_static_source"]["brightness"] == 10.0
    assert state["my_step_source"]["brightness"] == 15.0

    times = np.array([0.0, 1.5, 3.0])
    wavelengths = np.array([1000.0, 2000.0])

    values = model.evaluate(times, wavelengths, state)
    assert values.shape == (3, 2)
    assert np.allclose(values, [[10.0, 10.0], [25.0, 25.0], [10.0, 10.0]])


def test_additive_multi_source_node_resample() -> None:
    """Test that we can correctly resample a AdditiveMultiSourceModel."""
    ra = NumpyRandomFunc("uniform", low=10.0, high=20.0)
    dec = NumpyRandomFunc("uniform", low=-10.0, high=10.0)

    # Source1 and Source2 share the same ra and dec for each sample, but have
    # different brightness values.
    source1 = StaticSource(
        brightness=NumpyRandomFunc("uniform", low=10.0, high=20.0),
        ra=ra,
        dec=dec,
        node_label="my_static_source",
    )
    source2 = StepSource(
        brightness=NumpyRandomFunc("uniform", low=30.0, high=40.0),
        ra=ra,
        dec=dec,
        t0=1.0,
        t1=2.0,
        node_label="my_step_source",
    )
    model = AdditiveMultiSourceModel([source1, source2], node_label="my_multi_source")

    num_samples = 1000
    rng_info = np.random.default_rng(100)
    state = model.sample_parameters(num_samples=num_samples, rng_info=rng_info)
    assert len(np.unique(state["my_static_source"]["ra"])) > num_samples / 2
    assert len(np.unique(state["my_static_source"]["dec"])) > num_samples / 2
    assert np.allclose(state["my_static_source"]["ra"], state["my_static_source"]["ra"])
    assert np.allclose(state["my_static_source"]["dec"], state["my_static_source"]["dec"])

    brightness_diff = state["my_step_source"]["brightness"] - state["my_static_source"]["brightness"]
    assert np.all(brightness_diff > 5.0)

    # We can evaluate the combined model at all of the times.
    times = np.array([0.0, 1.5, 3.0])
    wavelengths = np.array([1000.0, 2000.0])

    values = model.evaluate(times, wavelengths, state)
    assert values.shape == (1000, 3, 2)

    assert np.allclose(values[:, 0, 0], state["my_static_source"]["brightness"])
    assert np.allclose(values[:, 0, 1], state["my_static_source"]["brightness"])
    assert np.allclose(
        values[:, 1, 0],
        state["my_static_source"]["brightness"] + state["my_step_source"]["brightness"],
    )
    assert np.allclose(
        values[:, 1, 1],
        state["my_static_source"]["brightness"] + state["my_step_source"]["brightness"],
    )
    assert np.allclose(values[:, 2, 0], state["my_static_source"]["brightness"])
    assert np.allclose(values[:, 2, 1], state["my_static_source"]["brightness"])


def test_additive_multi_source_node_redshift() -> None:
    """Test that we handle redshifts separately for each source."""
    source1 = StepSource(brightness=10.0, t0=1.0, t1=3.0, redshift=0.0, node_label="source1")
    source2 = StepSource(brightness=10.0, t0=2.0, t1=4.0, redshift=1.0, node_label="source2")
    model = AdditiveMultiSourceModel([source1, source2], node_label="my_multi_source")

    state = model.sample_parameters()
    times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    wavelengths = np.array([1000.0, 2000.0])

    # Compute the expected values:
    # The redshift applied to source1 is a no-op for both time and brightness.
    # The redshift applied to source2 shifts times to [1.25 1.75 2.25 2.75 3.25]
    # and scales up the brightness by a factor of (1 + z) = 2.0.
    contrib1 = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, 10.0], [0.0, 0.0], [0.0, 0.0]])
    contrib2 = np.array([[0.0, 0.0], [0.0, 0.0], [20.0, 20.0], [20.0, 20.0], [20.0, 20.0]])

    values = model.evaluate(times, wavelengths, state)
    assert values.shape == (5, 2)
    assert np.allclose(values, contrib1 + contrib2)


def test_additive_multi_source_node_effects_rest_frame() -> None:
    """Test that we handle rest frame effects separately for each source."""
    source1 = StepSource(brightness=10.0, t0=1.0, t1=3.0, node_label="source1")
    source2 = StepSource(brightness=10.0, t0=2.0, t1=4.0, node_label="source2")
    source1.add_effect(ConstantDimming(flux_fraction=0.5, rest_frame=True))
    source2.add_effect(ConstantDimming(flux_fraction=0.1, rest_frame=True))
    model = AdditiveMultiSourceModel([source1, source2], node_label="my_multi_source")

    # We added the effect to each submodel's rest frame list.
    assert len(source1.rest_frame_effects) == 1
    assert len(source2.rest_frame_effects) == 1
    assert len(source1.obs_frame_effects) == 0
    assert len(source2.obs_frame_effects) == 0
    assert len(model.rest_frame_effects) == 0
    assert len(model.obs_frame_effects) == 0

    state = model.sample_parameters()
    times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    wavelengths = np.array([1000.0, 2000.0])
    values = model.evaluate(times, wavelengths, state)
    assert values.shape == (5, 2)

    contrib1 = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0], [0.0, 0.0], [0.0, 0.0]])
    contrib2 = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    assert np.allclose(values, contrib1 + contrib2)


def test_additive_multi_source_node_effects_rest_frame_add() -> None:
    """Test that we handle rest frame effects separately for each source."""
    source1 = StepSource(brightness=10.0, t0=1.0, t1=3.0, node_label="source1")
    source2 = StepSource(brightness=10.0, t0=2.0, t1=4.0, node_label="source2")
    model = AdditiveMultiSourceModel([source1, source2], node_label="my_multi_source")
    model.add_effect(ConstantDimming(flux_fraction=0.5, rest_frame=True))

    # We added the effect to each submodel's rest frame list.
    assert len(source1.rest_frame_effects) == 1
    assert len(source2.rest_frame_effects) == 1
    assert len(source1.obs_frame_effects) == 0
    assert len(source2.obs_frame_effects) == 0
    assert len(model.rest_frame_effects) == 0
    assert len(model.obs_frame_effects) == 0

    state = model.sample_parameters()
    times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    wavelengths = np.array([1000.0, 2000.0])
    values = model.evaluate(times, wavelengths, state)
    assert values.shape == (5, 2)

    contrib1 = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0], [0.0, 0.0], [0.0, 0.0]])
    contrib2 = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0], [0.0, 0.0]])
    assert np.allclose(values, contrib1 + contrib2)


def test_additive_multi_source_node_effects_obs_frame() -> None:
    """Test that we handle observer frame effects together."""
    source1 = StepSource(brightness=10.0, t0=1.0, t1=3.0, node_label="source1")
    source2 = StepSource(brightness=10.0, t0=2.0, t1=4.0, node_label="source2")
    model = AdditiveMultiSourceModel([source1, source2], node_label="my_multi_source")
    model.add_effect(ConstantDimming(flux_fraction=0.5, rest_frame=False))

    # We added the effect to the joint model's rest frame list.
    assert len(source1.rest_frame_effects) == 0
    assert len(source2.rest_frame_effects) == 0
    assert len(source1.obs_frame_effects) == 0
    assert len(source2.obs_frame_effects) == 0
    assert len(model.rest_frame_effects) == 0
    assert len(model.obs_frame_effects) == 1

    state = model.sample_parameters()
    times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    wavelengths = np.array([1000.0, 2000.0])
    values = model.evaluate(times, wavelengths, state)
    assert values.shape == (5, 2)

    contrib1 = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, 5.0], [0.0, 0.0], [0.0, 0.0]])
    contrib2 = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0], [0.0, 0.0]])
    assert np.allclose(values, contrib1 + contrib2)


def test_additive_multi_source_node_effects_fail() -> None:
    """Test that we fail if any source includes an observer frame effect."""
    source1 = StepSource(brightness=10.0, t0=1.0, t1=3.0, node_label="source1")
    source2 = StepSource(brightness=10.0, t0=2.0, t1=4.0, node_label="source2")
    source2.add_effect(ConstantDimming(flux_fraction=0.5, rest_frame=False))

    with pytest.raises(ValueError):
        _ = AdditiveMultiSourceModel([source1, source2], node_label="my_multi_source")


def test_random_multi_source_node() -> None:
    """Test that we can create and evaluate a RandomMultiSourceModel."""
    source1 = StaticSource(brightness=10.0, node_label="source1")
    source2 = StaticSource(brightness=15.0, node_label="source2")
    model = RandomMultiSourceModel(
        [source1, source2],
        weights=[0.8, 0.2],
        node_label="my_multi_source",
    )

    state = model.sample_parameters(num_samples=10_000)
    assert np.all(state["source1"]["brightness"] == 10.0)
    assert np.all(state["source2"]["brightness"] == 15.0)

    # We should get approximately 80% of the samples from the first
    # source and 20% from the second source.
    source = np.array(state["my_multi_source"]["selected_source"], dtype=str)
    assert np.all((source == "source1") | (source == "source2"))
    assert np.sum(source == "source1") > 7000
    assert np.sum(source == "source2") > 1000

    # When we evaluate the model, we should get the expected values.
    times = np.array([0.0, 1.5, 3.0])
    wavelengths = np.array([1000.0, 2000.0])
    values = model.evaluate(times, wavelengths, state)

    assert values.shape == (10_000, 3, 2)
    assert np.all((values == 10.0) | (values == 15.0))
    assert np.sum(values == 10.0) > 7000 * 5
    assert np.sum(values == 15.0) > 1000 * 5
