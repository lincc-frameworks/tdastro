import numpy as np
from astropy import units as u
from tdastro.astro_utils.unit_utils import fnu_to_flam
from tdastro.math_nodes.np_random import NumpyRandomFunc
from tdastro.sources.sncomso_models import SncosmoWrapperModel


def test_sncomso_models_hsiao() -> None:
    """Test that we can create and evalue a 'hsiao' model."""
    model = SncosmoWrapperModel("hsiao", amplitude=2.0e10)
    state = model.sample_parameters()
    assert model.get_param(state, "amplitude") == 2.0e10
    assert model.get_param(state, "t0") == 0.0
    assert str(model) == "SncosmoWrapperModel_0"

    assert np.array_equal(model.param_names, ["amplitude"])
    assert np.array_equal(model.parameter_values, [2.0e10])

    # Test against a manual query from sncosmo with no redshift and an
    # unrealistically high amplitude.
    #     model = sncosmo.Model(source='hsiao')
    #     model.set(z=0.0, t0=0.0, amplitude=2.0e10)
    #     model.flux(5., [4000., 4100., 4200.])
    fluxes_fnu = model.evaluate([5.0], [4000.0, 4100.0, 4200.0])
    fluxes_flam = fnu_to_flam(
        fluxes_fnu,
        [4000.0, 4100.0, 4200.0],
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )
    assert np.allclose(fluxes_flam, [133.98143039, 152.74613574, 134.40916824])


def test_sncomso_models_hsiao_t0() -> None:
    """Test that we can create and evalue a 'hsiao' model with a t0."""
    model = SncosmoWrapperModel("hsiao", t0=55000.0, amplitude=2.0e10)
    state = model.sample_parameters()
    assert model.get_param(state, "amplitude") == 2.0e10
    assert model.get_param(state, "t0") == 55000.0

    assert np.array_equal(model.param_names, ["amplitude"])
    assert np.array_equal(model.parameter_values, [2.0e10])

    # Test against a manual query from sncosmo with no redshift and an unrealistically high amplitude.
    #     model = sncosmo.Model(source='hsiao')
    #     model.set(z=0.0, t0=55000., amplitude=2.0e10)
    #     model.flux(54990., [4000., 4100., 4200.])
    fluxes_fnu = model.evaluate([54990.0], [4000.0, 4100.0, 4200.0])
    fluxes_flam = fnu_to_flam(
        fluxes_fnu,
        [4000.0, 4100.0, 4200.0],
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )
    assert np.allclose(fluxes_flam, [67.83696271, 67.98471119, 47.20395186])


def test_sncomso_models_set() -> None:
    """Test that we can create and evalue a 'hsiao' model and set parameter."""
    model = SncosmoWrapperModel("hsiao", redshift=0.5)

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
