import numpy as np
from tdastro.sources.sncomso_models import SncosmoWrapperModel
from tdastro.util_nodes.np_random import NumpyRandomFunc


def test_sncomso_models_hsiao() -> None:
    """Test that we can create and evalue a 'hsiao' model."""
    model = SncosmoWrapperModel("hsiao", amplitude=1.0e-10)
    state = model.sample_parameters()
    assert model.get_param(state, "amplitude") == 1.0e-10
    assert str(model) == "0:tdastro.sources.sncomso_models.SncosmoWrapperModel"

    assert np.array_equal(model.param_names, ["amplitude"])
    assert np.array_equal(model.parameter_values, [1.0e-10])

    # Test with the example from: https://sncosmo.readthedocs.io/en/stable/models.html
    fluxes = model.evaluate([54990.0], [4000.0, 4100.0, 4200.0])
    assert np.allclose(fluxes, [4.31210900e-20, 7.46619962e-20, 1.42182787e-19])


def test_sncomso_models_set() -> None:
    """Test that we can create and evalue a 'hsiao' model and set parameter."""
    model = SncosmoWrapperModel("hsiao")

    # sncosmo parameters exist at default values.
    assert np.array_equal(model.param_names, ["amplitude"])
    assert np.array_equal(model.parameter_values, [1.0])

    model.set(amplitude=100.0)
    state = model.sample_parameters()

    assert model.get_param(state, "amplitude") == 100.0
    assert np.array_equal(model.param_names, ["amplitude"])
    assert np.array_equal(model.parameter_values, [100.0])


def test_sncomso_models_chained() -> None:
    """Test that we can create and evalue a 'hsiao' model using randomized parameters."""
    # Generate the amplitude from a uniform distribution, but use a fixed seed so we have
    # reproducible tests.
    model = SncosmoWrapperModel(
        "hsiao",
        amplitude=NumpyRandomFunc("uniform", low=2.0, high=12.0, seed=100),
    )
    _ = model.sample_parameters()

    assert np.array_equal(model.param_names, ["amplitude"])
    assert 2.0 <= model.parameter_values[0] <= 12.0

    # When we resample, we get different model parameters. Create a histogram
    # of 10,000 samples.
    hist = [0] * 10
    source_model = model.source
    for _ in range(10_000):
        _ = model.sample_parameters()
        bin = int(source_model.parameters[0] - 2.0)
        hist[bin] += 1

    # Each bin should have around 1,000 samples.
    for i in range(10):
        assert 800 < hist[i] < 1200
