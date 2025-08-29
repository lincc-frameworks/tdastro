"""Test the RedbackWrapperModel."""

from collections import namedtuple

import astropy.units as uu
import numpy as np
import pytest
from tdastro.astro_utils.unit_utils import fnu_to_flam
from tdastro.math_nodes.given_sampler import GivenValueList
from tdastro.sources.redback_models import RedbackWrapperModel


def toy_redback_model(times, lambda_array, t_start, slope, **kwargs):
    """A function simulating a toy redback model that produces known outputs.
    Uses a function that is linearly increasing in time for t=0 to 100.

    f(t, w) = { log(w) + (t-t_start) * slope    if t >= t_start
              { log(w)                          if t < t_start
    Parameters
    ----------
    times : np.ndarray
        This is not used by the model, which (consistent with other redback models queried
        for spectra) evaluates a predefined set of times.
    lambda_array : np.ndarray
        The wavelength values to use for the model (in nm).
    t_start : float
        The time offset to apply to the model. This is when the linear ramp starts.
    slope : float
        The increasing slope with time.
    **kwargs : dict
        Additional keyword arguments to pass to the model function.

    Returns
    -------
    A named tuple representing the modeled spectra of the data. The tuple has the
    following members:
    time : a T length array with the times at which the spectrum is evaluated
    lambdas : a W length array with the wavelengths (in nm) at which the spectrum is evaluated
    spectrum : a T x W sized matrix representing the modeled spectrum (in erg / cm^2 / s / Angstrom)
    """
    times_to_use = np.arange(0, 20)
    slope_addition = np.clip(times_to_use - t_start, 0, None) * slope
    spectra = lambda_array[np.newaxis, :] + slope_addition[:, np.newaxis]

    output = namedtuple("output", ["time", "lambdas", "spectra"])(
        time=times_to_use, lambdas=lambda_array, spectra=spectra
    )
    return output


def test_redback_models_toy() -> None:
    """Test that we can create and evaluate a toy model."""
    parameters = {
        "t_start": 5.0,
        "slope": 2.0,
        "redshift": 0.0,
    }
    model = RedbackWrapperModel(toy_redback_model, parameters=parameters, node_label="toy")
    assert np.isinf(model.maxwave())
    assert model.minwave() == 0.0
    assert set(model.source_param_names) == {"t_start", "slope", "redshift"}

    state = model.sample_parameters()
    assert state["toy"]["t_start"] == 5.0
    assert state["toy"]["slope"] == 2.0

    times = np.array([1.0, 5.0, 6.5, 10.0])
    waves_ang = np.array([1000.0, 2000.0])
    fluxes = model.evaluate(times, waves_ang, graph_state=state)

    expected_flam = np.array(
        [
            [100.0, 200.0],
            [100.0, 200.0],
            [103.0, 203.0],
            [110.0, 210.0],
        ]
    )

    # We need to convert the output back to flam to check them.
    output_flam = fnu_to_flam(
        fluxes,
        waves_ang,
        wave_unit=uu.AA,
        flam_unit=uu.erg / uu.second / uu.cm**2 / uu.AA,
        fnu_unit=uu.nJy,
    )
    assert np.allclose(output_flam, expected_flam)


def test_redback_models_toy_fail() -> None:
    """Test that we can create, but fail to evaluate a toy model if we don't have the parameters we need."""
    parameters = {
        "slope": 2.0,
        "redshift": 0.0,
    }
    model = RedbackWrapperModel(toy_redback_model, parameters=parameters, node_label="toy")

    state = model.sample_parameters()
    assert state["toy"]["slope"] == 2.0

    times = np.array([1.0, 5.0, 6.5, 10.0])
    waves_ang = np.array([1000.0, 2000.0])
    with pytest.raises(TypeError):
        _ = model.evaluate(times, waves_ang, graph_state=state)

    # Fail if we give it the same parameter in two different ways (repeat redshift).
    parameters["t_start"] = 1.0  # Fix missing paramater
    with pytest.raises(ValueError):
        _ = RedbackWrapperModel(toy_redback_model, parameters=parameters, redshift=0.1, node_label="toy")


def test_redback_models_toy_chained() -> None:
    """Test that we can create and evaluate a toy model with chained parameters."""
    parameters = {
        "t_start": GivenValueList([0.0, 2.0, 4.0]),
        "slope": 2.0,
        "redshift": 0.0,
    }
    model = RedbackWrapperModel(toy_redback_model, parameters=parameters, node_label="toy")

    state = model.sample_parameters(num_samples=3)
    times = np.array([0.0, 1.0, 5.0, 6.5, 10.0])
    waves_ang = np.array([1000.0])
    fluxes = model.evaluate(times, waves_ang, graph_state=state)

    # Each sample has the starting time of the ramp starting later.
    expected_flam = [
        np.array([100.0, 102.0, 110.0, 113.0, 120.0]),
        np.array([100.0, 100.0, 106.0, 109.0, 116.0]),
        np.array([100.0, 100.0, 102.0, 105.0, 112.0]),
    ]

    for idx in range(3):
        output_flam = fnu_to_flam(
            fluxes[idx, :],
            waves_ang,
            wave_unit=uu.AA,
            flam_unit=uu.erg / uu.second / uu.cm**2 / uu.AA,
            fnu_unit=uu.nJy,
        )
        assert np.allclose(output_flam[:, 0], expected_flam[idx])
