import numpy as np
import pytest
from tdastro.astro_utils.passbands import Passband, PassbandGroup
from tdastro.sources.lightcurve_source import LightcurveSource


def _create_toy_passbands() -> PassbandGroup:
    """Create a toy passband group with three passbands where the first passband
    has no overlap while the second two overlap each other for half the range.
    """
    a_band = Passband(np.array([[400, 0.5], [500, 0.5], [600, 0.5]]), "LSST", "u")
    b_band = Passband(np.array([[800, 0.8], [900, 0.8], [1000, 0.8]]), "LSST", "g")
    c_band = Passband(np.array([[900, 0.6], [1000, 0.6], [1100, 0.6]]), "LSST", "r")
    return PassbandGroup(given_passbands=[a_band, b_band, c_band])


def _create_toy_lightcurves() -> dict:
    """Create toy lightcurves where the first two are constant and the third
    is linearly increasing.  Each lightcurve covers a slightly different
    time range.
    """
    times = np.linspace(1, 11, 100)
    lightcurves = {
        "u": np.array([times - 0.2, 2.0 * np.ones_like(times)]).T,
        "g": np.array([times - 0.1, 3.0 * np.ones_like(times)]).T,
        "r": np.array([times, 0.1 * times + 1.0]).T,
    }
    return lightcurves


def test_create_lightcurve_source() -> None:
    """Test that we can create a simple LightcurveSource object."""
    pb_group = _create_toy_passbands()
    lightcurves = _create_toy_lightcurves()
    lc_source = LightcurveSource(lightcurves, pb_group)

    # Check the internal structure of the LightCurveSource.
    assert len(lc_source.lightcurves) == 3
    assert len(lc_source.sed_values) == 3
    assert np.allclose(lc_source.all_waves, pb_group.waves)

    filters = list(lc_source.lightcurves.keys())
    assert filters == ["u", "g", "r"]

    # Check that no two SED basis functions overlap.
    for f1 in filters:
        for f2 in filters:
            if f1 != f2:
                assert np.count_nonzero(lc_source.sed_values[f1] * lc_source.sed_values[f2]) == 0

    # A call to get_band_fluxes should return the desired lightcurves.  We only use two of the passbands.
    graph_state = lc_source.sample_parameters(num_samples=1)  # dummy. unused.
    query_times = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 20.0, 21.0])
    query_filters = np.array(["u", "r", "u", "r", "u", "r", "u", "r"])
    fluxes = lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert len(fluxes) == len(query_times)

    # Timesteps 0.0, 0.5, 20.0, and 21.0 fall outside the range of the model and are set to 0.0.
    # Timesteps 1.0 and 3.0 are from the y band which is constant at 2.
    # Timesteps 2.0 and 4.0 are from the r band which is linearly increasing with time.
    assert np.allclose(fluxes, [0.0, 0.0, 2.0, 1.2, 2.0, 1.4, 0.0, 0.0])


def test_lightcurve_source_nonoverlap() -> None:
    """Test that we can query the LightcurveSource with wavelengths that
    do not overlap the lightcurves.
    """
    pb_group = _create_toy_passbands()
    lightcurves = _create_toy_lightcurves()
    del lightcurves["u"]  # Remove the u band lightcurve.

    lc_source = LightcurveSource(lightcurves, pb_group)
    assert len(lc_source.lightcurves) == 2
    assert len(lc_source.sed_values) == 2
    assert np.allclose(lc_source.all_waves, pb_group.waves)

    filters = list(lc_source.lightcurves.keys())
    assert filters == ["g", "r"]

    # A call to get_band_fluxes should return the desired lightcurves.  We query
    # u and g bands. Since u is not present, we should get zeros at those times.
    graph_state = lc_source.sample_parameters(num_samples=1)  # dummy. unused.
    query_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    query_filters = np.array(["g", "u", "g", "u", "g", "u", "g"])
    fluxes = lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert len(fluxes) == len(query_times)

    assert np.allclose(fluxes, [0.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0])


def test_create_lightcurve_source_fail() -> None:
    """Test fail cases for creating the LightCurveSource object."""
    a_band = Passband(np.array([[400, 0.5], [500, 0.5], [600, 0.5]]), "LSST", "u")
    b_band = Passband(np.array([[800, 0.8], [900, 0.8], [1000, 0.8]]), "LSST", "g")
    c_band = Passband(np.array([[900, 0.6], [1000, 0.6], [1100, 0.6]]), "LSST", "r")
    pb_group = PassbandGroup(given_passbands=[a_band, b_band, c_band])

    times = np.linspace(0, 10, 100)
    lightcurves = {
        "u": np.array([times, 2.0 * np.ones_like(times)]).T,
        "g": np.array([times, 3.0 * np.ones_like(times)]).T,
        "r": np.array([times, 4.0 * np.ones_like(times)]).T,
        "i": np.array([times, 0.1 * times + 1.0]).T,
    }

    # Fail on mismatched passbands.
    with pytest.raises(ValueError):
        _ = LightcurveSource(lightcurves, pb_group)

    # Remove the offending passband and try again.
    del lightcurves["i"]
    _ = LightcurveSource(lightcurves, pb_group)

    # Make one of the lightcurves the wrong shape.
    lightcurves["u"] = times.T
    with pytest.raises(ValueError):
        _ = LightcurveSource(lightcurves, pb_group)
    lightcurves["u"] = np.array([times, 2.0 * np.ones_like(times)]).T

    # We fail if two passbands overlap other passbands completely.
    d_band = Passband(np.array([[850, 0.6], [1050, 0.6]]), "LSST", "i")
    pb_group = PassbandGroup(given_passbands=[a_band, b_band, c_band, d_band])
    lightcurves["i"] = np.array([times, 0.1 * times + 1.0]).T
    with pytest.raises(ValueError):
        _ = LightcurveSource(lightcurves, pb_group)


def test_lightcurve_plot() -> None:
    """Test that the plotting functions do not crash."""
    pb_group = _create_toy_passbands()
    lightcurves = _create_toy_lightcurves()
    lc_source = LightcurveSource(lightcurves, pb_group)
    lc_source.plot_lightcurves()
    lc_source.plot_sed_basis()
