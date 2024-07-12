import numpy as np
import pytest
from tdastro.effects.white_noise import WhiteNoise
from tdastro.populations.fixed_population import FixedPopulation
from tdastro.sources.static_source import StaticSource


def test_fixed_population_basic_add():
    """Test that we can add effects to a population of PhysicalModels."""
    population = FixedPopulation()
    population.add_source(StaticSource(brightness=10.0), 0.5)
    assert population.num_sources == 1
    assert np.allclose(population.probs, [1.0])

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


def test_fixed_population_sample():
    """Test that we can sample and create a StaticSource object."""
    test_rng = np.random.default_rng(100)
    population = FixedPopulation(rng=test_rng)

    population.add_source(StaticSource(brightness=0.0), 10.0)
    assert np.allclose(population.probs, [1.0])
    assert population.num_sources == 1

    population.add_source(StaticSource(brightness=1.0), 10.0)
    assert np.allclose(population.probs, [0.5, 0.5])
    assert population.num_sources == 2

    population.add_source(StaticSource(brightness=2.0), 20.0)
    assert np.allclose(population.probs, [0.25, 0.25, 0.5])
    assert population.num_sources == 3

    itr = 10_000
    counts = [0.0, 0.0, 0.0]
    for _ in range(itr):
        model = population.draw_source()
        counts[int(model.brightness)] += 1.0
    assert np.allclose(counts, [0.25 * itr, 0.25 * itr, 0.5 * itr], rtol=0.05)

    # Check the we can change a rate.
    population.change_rate(0, 20.0)
    assert np.allclose(population.probs, [0.4, 0.2, 0.4])

    counts = [0.0, 0.0, 0.0]
    for _ in range(itr):
        model = population.draw_source()
        counts[int(model.brightness)] += 1.0
    assert np.allclose(counts, [0.4 * itr, 0.2 * itr, 0.4 * itr], rtol=0.05)
