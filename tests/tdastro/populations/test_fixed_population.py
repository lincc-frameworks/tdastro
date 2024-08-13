import random

import numpy as np
import pytest
from tdastro.base_models import FunctionNode
from tdastro.effects.white_noise import WhiteNoise
from tdastro.populations.fixed_population import FixedPopulation
from tdastro.sources.static_source import StaticSource


def test_fixed_population_basic_add():
    """Test that we can add effects to a population of PhysicalModels."""
    population = FixedPopulation()
    population.add_source(StaticSource(brightness=10.0), 0.5)
    assert population.num_sources == 1
    assert np.allclose(population.weights, [0.5])

    # Test that we fail with a bad rate.
    with pytest.raises(ValueError):
        population.add_source(StaticSource(brightness=10.0), -0.5)


def test_fixed_population_add_effect():
    """Test that we can add effects to a population of PhysicalModels."""
    model1 = StaticSource(brightness=10.0)
    model2 = StaticSource(brightness=20.0)

    population = FixedPopulation()
    population.add_source(model1, 0.5)
    population.add_source(model2, 0.5)
    assert population.num_sources == 2
    assert len(model1.effects) == 0
    assert len(model2.effects) == 0

    # Add a white noise effect to all models.
    population.add_effect(WhiteNoise(scale=0.01))
    assert len(model1.effects) == 1
    assert len(model2.effects) == 1


def test_fixed_population_add_effect_fail():
    """Test a case where we try to add an existing effect to models."""
    model1 = StaticSource(brightness=10.0)
    model1.add_effect(WhiteNoise(scale=0.01))

    population = FixedPopulation()
    population.add_source(model1, 0.5)

    # Fail when we try to re-add the WhiteNoise effect
    with pytest.raises(ValueError):
        population.add_effect(WhiteNoise(scale=0.01))


def test_fixed_population_sample_sources():
    """Test that we can create a population of sources and sample its sources."""
    random.seed(1000)
    population = FixedPopulation()

    population.add_source(StaticSource(brightness=0.0), 10.0)
    assert np.allclose(population.weights, [10.0])
    assert population.num_sources == 1

    population.add_source(StaticSource(brightness=1.0), 10.0)
    assert np.allclose(population.weights, [10.0, 10.0])
    assert population.num_sources == 2

    population.add_source(StaticSource(brightness=2.0), 20.0)
    assert np.allclose(population.weights, [10.0, 10.0, 20.0])
    assert population.num_sources == 3

    itr = 10_000
    counts = [0.0, 0.0, 0.0]
    for _ in range(itr):
        model = population.draw_source()
        state = model.sample_parameters()
        counts[int(model.get_param(state, "brightness"))] += 1.0
    assert np.allclose(counts, [0.25 * itr, 0.25 * itr, 0.5 * itr], rtol=0.05)

    # Check the we can change a rate.
    population.change_rate(0, 20.0)

    counts = [0.0, 0.0, 0.0]
    for _ in range(itr):
        model = population.draw_source()
        state = model.sample_parameters()
        counts[int(model.get_param(state, "brightness"))] += 1.0
    assert np.allclose(counts, [0.4 * itr, 0.2 * itr, 0.4 * itr], rtol=0.05)


def test_fixed_population_sample_fluxes():
    """Test that we can create a population of sources and sample its sources' flux."""
    random.seed(1001)
    brightness_func = FunctionNode(random.uniform, a=0.0, b=100.0)
    population = FixedPopulation()
    population.add_source(StaticSource(brightness=100.0), 10.0)
    population.add_source(StaticSource(brightness=brightness_func), 10.0)
    population.add_source(StaticSource(brightness=200.0), 20.0)
    population.add_source(StaticSource(brightness=150.0), 10.0)

    # Sample the actual observations, resampling the corresponding
    # model's parameters each time.
    num_samples = 10_000
    times = np.array([1, 2, 3, 4, 5])
    wavelengths = np.array([100.0, 200.0, 300.0])
    fluxes = population.evaluate(num_samples, times, wavelengths)

    # Check that the fluxes are constant within a sample. Also check that we have
    # More than 4 values (since we are resampling a model with a random parameter).
    seen_values = []
    for i in range(num_samples):
        value = fluxes[i, 0, 0]
        seen_values.append(value)
        assert np.allclose(fluxes[i], value)
    assert len(np.unique(seen_values)) > 4

    # Check that our average is near the expected value of flux.
    ave_val = np.mean(fluxes.flatten())
    expected = 0.2 * 100.0 + 0.2 * 50.0 + 0.4 * 200.0 + 0.2 * 150.0
    assert abs(ave_val - expected) < 10.0
