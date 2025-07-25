import numpy as np
import pytest
from astropy.table import Table
from tdastro.astro_utils.mag_flux import mag2flux
from tdastro.astro_utils.passbands import Passband, PassbandGroup
from tdastro.effects.basic_effects import ConstantDimming
from tdastro.sources.lightcurve_source import (
    LightcurveData,
    LightcurveSource,
    MultiLightcurveSource,
)


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
    times = np.linspace(1, 11, 20)
    lightcurves = {
        "u": np.array([times - 0.2, 2.0 * np.ones_like(times)]).T,
        "g": np.array([times - 0.1, 3.0 * np.ones_like(times)]).T,
        "r": np.array([times, 0.1 * times + 1.0]).T,
    }
    return lightcurves


def test_create_lightcurve_data_from_dict() -> None:
    """Test that we can create a simple LightcurveData object from a dict."""
    lightcurves = _create_toy_lightcurves()
    lc_data = LightcurveData(lightcurves)

    # Check the internal structure of the LightcurveData.
    assert len(lc_data) == 3
    assert lc_data.lc_t0 == 0.0
    assert lc_data.period is None
    assert lc_data.filters == ["u", "g", "r"]
    for filt in ["u", "g", "r"]:
        assert np.allclose(lc_data.lightcurves[filt], lightcurves[filt])
        assert lc_data.baseline[filt] == 0.0

    # If we use an lc_t0, we should shift the lightcurves' times accordingly
    # and provide a baseline.
    lc_data2 = LightcurveData(lightcurves, lc_t0=2.0, baseline={"u": 0.1, "g": 0.2, "r": 0.3})
    assert lc_data2.lc_t0 == 2.0
    assert lc_data2.period is None
    for filt in ["u", "g", "r"]:
        assert np.allclose(lc_data2.lightcurves[filt][:, 0], lightcurves[filt][:, 0] - 2.0)
        assert np.allclose(lc_data2.lightcurves[filt][:, 1], lightcurves[filt][:, 1])
    assert lc_data2.baseline == {"u": 0.1, "g": 0.2, "r": 0.3}

    # We fail if the baseline does not match the filters.
    with pytest.raises(ValueError):
        _ = LightcurveData(lightcurves, lc_t0=2.0, baseline={"u": 0.1, "g": 0.2, "i": 0.3})

    # If we mark them as periodic we should fail because the times do not match.
    with pytest.raises(ValueError):
        _ = LightcurveData(lightcurves, periodic=True)


def test_create_lightcurve_data_periodic_from_dict() -> None:
    """Test that we can create a periodic LightcurveData object from a dict."""
    times = np.linspace(3, 13, 20)
    lightcurves = {
        "u": np.array([times, 2.0 * np.ones_like(times)]).T,
        "g": np.array([times, 3.0 * np.ones_like(times)]).T,
    }
    lc_data = LightcurveData(lightcurves, periodic=True)

    assert len(lc_data) == 2
    assert lc_data.filters == ["u", "g"]
    assert lc_data.lc_t0 == 3.0
    assert lc_data.period == 10.0

    # The values should be the same and the times should be shifted by 3.0.
    for filt in ["u", "g"]:
        assert np.allclose(lc_data.lightcurves[filt][:, 0], lightcurves[filt][:, 0] - 3.0)
        assert np.allclose(lc_data.lightcurves[filt][:, 1], lightcurves[filt][:, 1])
    assert lc_data.baseline == {"u": 0.0, "g": 0.0}


def test_create_lightcurve_data_from_numpy() -> None:
    """Test that we can create a simple LightcurveData object from a numpy array."""
    lightcurves = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [1.0, 2.0, 1.0, 1.5, 2.0, 2.5, 1.5, 1.0, 0.0, 1.0],
            ["u", "g", "r", "u", "g", "r", "u", "g", "r", "u"],
        ]
    ).T
    lc_data = LightcurveData(lightcurves)

    # Check the internal structure of the LightcurveData.
    assert len(lc_data) == 3
    assert lc_data.lc_t0 == 0.0
    assert lc_data.period is None
    assert set(lc_data.filters) == {"u", "g", "r"}
    assert np.allclose(lc_data.lightcurves["u"][:, 0], [0.0, 3.0, 6.0, 9.0])
    assert np.allclose(lc_data.lightcurves["u"][:, 1], [1.0, 1.5, 1.5, 1.0])
    assert np.allclose(lc_data.lightcurves["g"][:, 0], [1.0, 4.0, 7.0])
    assert np.allclose(lc_data.lightcurves["g"][:, 1], [2.0, 2.0, 1.0])
    assert np.allclose(lc_data.lightcurves["r"][:, 0], [2.0, 5.0, 8.0])
    assert np.allclose(lc_data.lightcurves["r"][:, 1], [1.0, 2.5, 0.0])
    assert lc_data.baseline == {"u": 0.0, "g": 0.0, "r": 0.0}


def test_create_lightcurve_data_from_lclib_table() -> None:
    """Test that we can create a simple LightcurveData object from a LCLIB table."""
    # When creating the LightcurveData from a table, we specifc the values in magnitude,
    # instead of flux.
    data = {
        "time": [0.0, 1.0, 2.0, 3.0, 4.0, 0.0],
        "type": ["S", "S", "S", "S", "S", "T"],
        "u": [1.0, 2.0, 1.0, 1.5, 2.0, 1.0],
        "g": [2.0, 3.0, 2.0, 2.5, 3.0, 2.0],
        "r": [1.0, 1.5, 2.0, 2.5, 3.0, 2.0],
        "i": [0.5, 0.6, 0.7, 0.8, 0.9, 1.5],
    }
    table = Table(data)
    table.meta["RECUR_CLASS"] = "NON-RECUR"
    lc_data = LightcurveData.from_lclib_table(table)

    # Check the internal structure of the LightcurveData.
    assert len(lc_data) == 4
    assert lc_data.lc_t0 == 0.0
    assert lc_data.period is None
    assert lc_data.filters == ["u", "g", "r", "i"]

    for filt in ["u", "g", "r", "i"]:
        assert np.allclose(lc_data.lightcurves[filt][:, 0], data["time"][:5])
        assert np.allclose(
            lc_data.lightcurves[filt][:, 1],
            mag2flux(np.array(data[filt][:5])),
        )

    # We get the baseline from the "T" type row.
    print(lc_data.baseline)
    assert lc_data.baseline["u"] == 1.0
    assert lc_data.baseline["g"] == 2.0
    assert lc_data.baseline["r"] == 2.0
    assert lc_data.baseline["i"] == 1.5

    # If we specify a subset of filters, only load those.
    lc_data = LightcurveData.from_lclib_table(table, filters=["u", "g"])
    assert set(lc_data.filters) == set(["u", "g"])
    assert "r" not in lc_data.lightcurves
    assert "i" not in lc_data.lightcurves
    assert "u" in lc_data.lightcurves
    assert "g" in lc_data.lightcurves


def test_create_lightcurve_data_from_lclib_table_periodic() -> None:
    """Test that we can create a periodic LightcurveData object from a LCLIB table."""
    # When creating the LightcurveData from a table, we specifc the values in magnitude,
    # instead of flux.
    data = {
        "time": [0.0, 1.0, 2.0, 3.0, 4.0],
        "type": ["S", "S", "S", "S", "S"],
        "u": [1.0, 2.0, 1.0, 1.5, 1.0],
        "g": [2.0, 3.0, 2.0, 2.5, 2.0],
        "i": [1.0, 2.0, 3.0, 4.0, 5.0],  # First does not match last.
    }
    table = Table(data)
    table.meta["RECUR_CLASS"] = "RECUR-PERIODIC"
    lc_data = LightcurveData.from_lclib_table(table)

    # Check the internal structure of the LightcurveData.
    assert len(lc_data) == 3
    assert lc_data.lc_t0 == 0.0
    assert np.allclose(lc_data.period, 5.0)
    assert lc_data.filters == ["u", "g", "i"]

    # The first 5 entries match the entry data and the last was inserted
    # to match the periodicity.
    for filt in ["u", "g", "i"]:
        assert len(lc_data.lightcurves[filt]) == 6
        assert np.allclose(lc_data.lightcurves[filt][:5, 0], data["time"])
        assert np.allclose(lc_data.lightcurves[filt][5, 0], 5.0)

        assert np.allclose(
            lc_data.lightcurves[filt][:5, 1],
            mag2flux(np.array(data[filt])),
        )
        assert np.allclose(
            lc_data.lightcurves[filt][5, 1],
            lc_data.lightcurves[filt][0, 1],
        )


def test_create_lightcurve_source() -> None:
    """Test that we can create a simple LightcurveSource object."""
    pb_group = _create_toy_passbands()
    lightcurves = _create_toy_lightcurves()
    lc_source = LightcurveSource(lightcurves, pb_group, t0=0.0)

    # Check the internal structure of the LightCurveSource.
    assert len(lc_source.lightcurves) == 3
    assert len(lc_source.sed_values) == 3
    assert np.allclose(lc_source.all_waves, pb_group.waves)
    assert lc_source.filters == ["u", "g", "r"]
    assert lc_source.apply_redshift is False

    # Check that we fail if we try to turn on redshift.
    with pytest.raises(NotImplementedError):
        lc_source.set_apply_redshift(True)

    # Check that no two SED basis functions overlap.
    for f1 in lc_source.filters:
        for f2 in lc_source.filters:
            if f1 != f2:
                assert np.count_nonzero(lc_source.sed_values[f1] * lc_source.sed_values[f2]) == 0

    # A call to get_band_fluxes should return the desired lightcurves.  We only use two of the passbands.
    graph_state = lc_source.sample_parameters(num_samples=1)
    query_times = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 20.0, 21.0])
    query_filters = np.array(["u", "r", "u", "r", "u", "r", "u", "r"])
    fluxes = lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert len(fluxes) == len(query_times)

    # Timesteps 0.0, 0.5, 20.0, and 21.0 fall outside the range of the model and are set to 0.0.
    # Timesteps 1.0 and 3.0 are from the u band which is constant at 2.
    # Timesteps 2.0 and 4.0 are from the r band which is linearly increasing with time.
    assert np.allclose(fluxes, [0.0, 0.0, 2.0, 1.2, 2.0, 1.4, 0.0, 0.0])

    # A call to get_band_fluxes with an unsupported filter should raise an error.
    query_filters = np.array(["u", "r", "u", "r", "u", "x", "u", "r", "r"])
    with pytest.raises(ValueError):
        lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)


def test_create_lightcurve_source_unsorted() -> None:
    """Test that we fail if we try to create a LightcurveSource with unsorted lightcurves."""
    pb_group = _create_toy_passbands()
    lightcurves = _create_toy_lightcurves()
    lightcurves["u"][0, 0] = 2.0  # Make the u band unsorted

    with pytest.raises(ValueError):
        # We should fail because the lightcurves are not sorted by time.
        LightcurveSource(lightcurves, pb_group, t0=0.0)


def test_create_lightcurve_source_baseline() -> None:
    """Test that we can create a simple LightcurveSource object with baseline values."""
    pb_group = _create_toy_passbands()
    lightcurves = _create_toy_lightcurves()
    baseline = {"u": 0.5, "g": 1.2, "r": 0.05}
    lc_source = LightcurveSource(lightcurves, pb_group, t0=0.0, baseline=baseline)

    # A call to get_band_fluxes should return the desired lightcurves.  We only use two of the passbands.
    graph_state = lc_source.sample_parameters(num_samples=1)
    query_times = np.array([-100.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 20.0, 21.0, 1000.0])
    query_filters = np.array(["u", "u", "r", "u", "r", "u", "r", "u", "r", "r"])
    fluxes = lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert len(fluxes) == len(query_times)

    # Timesteps -100.0, 0.0, 0.5, 20.0, 21.0, 1000.0 all fall outside the range of the model
    # and use baseline values for the respective bands.
    # Timesteps 1.0 and 3.0 are from the u band which is constant at 2.
    # Timesteps 2.0 and 4.0 are from the r band which is linearly increasing with time.
    assert np.allclose(fluxes, [0.5, 0.5, 0.05, 2.0, 1.2, 2.0, 1.4, 0.5, 0.05, 0.05])

    # We fail if we try to create a LightcurveSource with a baseline that does
    # not match the passbands (no r band provided).
    with pytest.raises(ValueError):
        LightcurveSource(lightcurves, pb_group, t0=0.0, baseline={"u": 0.5, "g": 1.2})

    # Test that we can add a constant dimming effect and it is applied in to the bandpass values.
    effect = ConstantDimming(flux_fraction=0.1)
    lc_source.add_effect(effect)
    graph_state = lc_source.sample_parameters(num_samples=1)

    fluxes = lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert len(fluxes) == len(query_times)
    assert np.allclose(fluxes, [0.05, 0.05, 0.005, 0.20, 0.12, 0.20, 0.14, 0.05, 0.005, 0.005])


def test_create_lightcurve_source_periodic() -> None:
    """Test that we can create a periodic LightcurveSource object."""
    pb_group = _create_toy_passbands()

    with pytest.raises(ValueError):
        # We cannot create a periodic lightcurve source lightcurves that do
        # not cover the same time range.
        lightcurves = _create_toy_lightcurves()
        LightcurveSource(lightcurves, pb_group, periodic=True)

    times = np.arange(0.0, 10.5, 0.5)
    g_curve = 3.0 * np.ones_like(times)
    r_curve = 0.1 * times + 1.0
    r_curve[-1] = r_curve[0]  # Make sure the 1st and last values are the same
    lightcurves = {
        "g": np.array([times, g_curve]).T,
        "r": np.array([times, r_curve]).T,
    }
    lc_source = LightcurveSource(lightcurves, pb_group, t0=0.0, periodic=True)

    # A call to get_band_fluxes should return the desired lightcurves.
    graph_state = lc_source.sample_parameters(num_samples=1)
    query_times = np.array([1.0, 5.0, 11.0, 15.0, 21.0, 25.0, 51.0])
    query_filters = np.full(len(query_times), "r")
    fluxes = lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert len(fluxes) == len(query_times)

    # Time steps 1.0, 11.0, 21.0, and 51.0 are all wrapped to the same point.
    # Time steps 5.0, 15.0, and 25.0 are all wrapped to the same point.
    assert np.allclose(fluxes, [1.1, 1.5, 1.1, 1.5, 1.1, 1.5, 1.1])

    # Check a curve if defined with a first time > 0.0.
    times = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    g_curve = 3.0 * np.ones_like(times)
    r_curve = [0.0, 1.0, 2.0, 3.0, 2.5, 1.5, 0.0]
    lightcurves = {
        "g": np.array([times, g_curve]).T,
        "r": np.array([times, r_curve]).T,
    }

    # We fail if we specify an incorrect lc_t0 for a periodic lightcurve.
    with pytest.raises(ValueError):
        _ = LightcurveSource(lightcurves, pb_group, lc_t0=1.0, t0=0.0, periodic=True)

    lc_source = LightcurveSource(lightcurves, pb_group, lc_t0=2.0, t0=0.0, periodic=True)
    query_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    query_filters = np.full(len(query_times), "r")

    # A call to get_band_fluxes should return the desired lightcurves. Since t0=0.0, query
    # time 0.0 should correspond to the start of the lightcurve's period.
    graph_state = lc_source.sample_parameters(num_samples=1)
    fluxes = lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert np.allclose(fluxes, [0.0, 1.0, 2.0, 3.0, 2.5])

    # We can also auto-derive lc_t0 from the lightcurves.
    lc_source = LightcurveSource(lightcurves, pb_group, t0=0.0, periodic=True)
    graph_state = lc_source.sample_parameters(num_samples=1)
    fluxes = lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert np.allclose(fluxes, [0.0, 1.0, 2.0, 3.0, 2.5])

    # If we use t0=1.0, we are saying the period starts at 1.0 for this sample, so a
    # query time of 0.0 should wrap around and return the *last* value of the lightcurve.
    lc_source = LightcurveSource(lightcurves, pb_group, t0=1.0, periodic=True)
    graph_state = lc_source.sample_parameters(num_samples=1)
    fluxes = lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert np.allclose(fluxes, [1.5, 0.0, 1.0, 2.0, 3.0])


def test_create_lightcurve_source_periodic_complex_offsets() -> None:
    """Test that we can create a periodic LightcurveSource with complex offsets."""
    pb_group = _create_toy_passbands()

    # Create a lightcurve with 20 samples over a 10 day time range.
    dt = np.arange(0.0, 10.5, 0.5)
    r_curve = np.abs(dt - 5.0)  # Sawtooth-like curve starting and ending at 5.0
    g_curve = 3.0 * np.ones_like(dt)

    # Define the actually times for the lightcurves as starting at 60676.0.
    times = 60676.0 + dt
    lightcurves = {
        "g": np.array([times, g_curve]).T,
        "r": np.array([times, r_curve]).T,
    }

    # Create a LightcurveSource with t0=60672.0, so we are shifting it back by 4 days.
    lc_source = LightcurveSource(lightcurves, pb_group, t0=60672.0, periodic=True)
    graph_state = lc_source.sample_parameters(num_samples=1)

    # Check query times relative to 60676.0 (4 days after the period started).
    query_times = 60676.0 + np.array([0.0, 0.5, 1.0, 2.0, 6.5, 12.0])
    query_filters = np.full(len(query_times), "r")
    fluxes = lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert np.allclose(fluxes, [1.0, 0.5, 0.0, 1.0, 4.5, 1.0])


def test_create_lightcurve_source_numpy() -> None:
    """Test that we can create a simple LightcurveSource object from a numpy array."""
    pb_group = _create_toy_passbands()

    # Create fake lightcurves over the time range 0.0 to 4.3. The u band is linerly
    # decreasing, the g band is constant, and the r band is linearly increasing.
    lightcurves = np.array(
        [
            [0.0, 10.0, "u"],
            [0.1, 11.0, "g"],
            [0.3, 11.0, "r"],
            [1.0, 10.1, "u"],
            [1.1, 11.0, "g"],
            [1.3, 10.9, "r"],
            [2.0, 10.2, "u"],
            [2.1, 11.0, "g"],
            [2.3, 10.8, "r"],
            [3.0, 10.3, "u"],
            [3.1, 11.0, "g"],
            [3.3, 10.7, "r"],
            [4.0, 10.4, "u"],
            [4.1, 11.0, "g"],
            [4.3, 10.6, "r"],
        ]
    )
    lc_source = LightcurveSource(lightcurves, pb_group, t0=0.0)

    # Check the internal structure of the LightCurveSource.
    assert len(lc_source.lightcurves) == 3
    assert len(lc_source.sed_values) == 3
    assert np.allclose(lc_source.all_waves, pb_group.waves)
    assert set(lc_source.filters) == set(["u", "g", "r"])

    # A call to get_band_fluxes should return the desired lightcurves.
    graph_state = lc_source.sample_parameters(num_samples=1)
    query_times = np.array([0.5, 0.6, 1.8, 2.3, 2.8, 3.0, 3.5, 4.0])
    query_filters = np.array(["u", "g", "r", "r", "r", "g", "u", "u"])
    expected = np.array([10.05, 11.0, 10.85, 10.8, 10.75, 11.0, 10.35, 10.4])

    fluxes = lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert len(fluxes) == len(expected)
    assert np.allclose(fluxes, expected)


def test_create_lightcurve_source_t0() -> None:
    """Test that we can create a simple LightcurveSource object with a given t0."""
    pb_group = _create_toy_passbands()
    lightcurves = _create_toy_lightcurves()
    lc_source = LightcurveSource(lightcurves, pb_group, t0=60676.0)

    graph_state = lc_source.sample_parameters(num_samples=1)  # needed for t0
    query_times = 60676.0 + np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 20.0, 21.0])
    query_filters = np.array(["u", "r", "u", "r", "u", "r", "u", "r"])
    fluxes = lc_source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert len(fluxes) == len(query_times)

    # Timesteps +0.0, +0.5, +20.0, and +21.0 fall outside the range of the model and are set to 0.0.
    # Timesteps +1.0 and +3.0 are from the u band which is constant at 2.
    # Timesteps +2.0 and +4.0 are from the r band which is linearly increasing with time
    # as 0.1 * t + 1.0.
    assert np.allclose(fluxes, [0.0, 0.0, 2.0, 1.2, 2.0, 1.4, 0.0, 0.0])

    # Test that we can also handle a lightcurve with a different lc_t0.
    lc_source2 = LightcurveSource(lightcurves, pb_group, t0=60676.0, lc_t0=1.0)
    graph_state2 = lc_source2.sample_parameters(num_samples=1)  # needed for t0
    fluxes2 = lc_source2.get_band_fluxes(pb_group, query_times, query_filters, graph_state2)

    # Timesteps +20.0 and +21.0 fall outside the range of the model and are set to 0.0.
    # Timesteps +0.0, +1.0 and +3.0 correspond to the original times (in the lightcurve definition)
    # of +1.0, +2.0, and +3.0 which are from the u band which is constant at 2.
    # Timesteps +0.5, +2.0 and +4.0 correspond to the original times (in the lightcurve definition)
    # of +1.5, +2.0, and +5.0 which are from the r band which is linearly increasing with time
    # as 0.1 * t + 1.0.
    assert np.allclose(fluxes2, [2.0, 1.15, 2.0, 1.3, 2.0, 1.5, 0.0, 0.0])


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
        _ = LightcurveSource(lightcurves, pb_group, t0=0.0)

    # Remove the offending passband and try again.
    del lightcurves["i"]
    _ = LightcurveSource(lightcurves, pb_group, t0=0.0)

    # We fail without a t0 value.
    with pytest.raises(ValueError):
        _ = LightcurveSource(lightcurves, pb_group)

    # Make one of the lightcurves the wrong shape.
    lightcurves["u"] = times.T
    with pytest.raises(ValueError):
        _ = LightcurveSource(lightcurves, pb_group, t0=0.0)
    lightcurves["u"] = np.array([times, 2.0 * np.ones_like(times)]).T

    # We fail if two passbands overlap other passbands completely.
    d_band = Passband(np.array([[850, 0.6], [1050, 0.6]]), "LSST", "i")
    pb_group = PassbandGroup(given_passbands=[a_band, b_band, c_band, d_band])
    lightcurves["i"] = np.array([times, 0.1 * times + 1.0]).T
    with pytest.raises(ValueError):
        _ = LightcurveSource(lightcurves, pb_group, t0=0.0)


def test_lightcurve_plot() -> None:
    """Test that the plotting functions do not crash."""
    pb_group = _create_toy_passbands()
    lightcurves = _create_toy_lightcurves()
    lc_source = LightcurveSource(lightcurves, pb_group, t0=0.0)
    lc_source.plot_lightcurves()
    lc_source.plot_sed_basis()


def test_create_multilightcurve_source() -> None:
    """Test that we can create a simple MultiLightcurveSource object."""
    pb_group = _create_toy_passbands()

    # Lightcurve 1 is non-periodic and covers u and g.
    lc1_times = np.arange(0.0, 10.5, 0.5)
    lc1_lightcurves = {
        "u": np.array([lc1_times + 0.1, 2.0 * np.ones_like(lc1_times)]).T,
        "g": np.array([lc1_times, 3.0 * np.ones_like(lc1_times)]).T,
    }
    lc1_data = LightcurveData(lc1_lightcurves, lc_t0=0.0, baseline={"u": 0.1, "g": 0.2})

    # Lightcurve 2 is periodic and covers r and g.
    lc2_times = np.arange(0.0, 19.0, 1.0)
    lc2_lightcurves = {
        "r": np.array([lc2_times, lc2_times % 2]).T,
        "g": np.array([lc2_times, lc2_times % 2 + 0.5]).T,
    }
    lc2_data = LightcurveData(lc2_lightcurves, periodic=True)

    # Create the MultiLightcurveSource with both lightcurves.
    source = MultiLightcurveSource(
        [lc1_data, lc2_data],
        pb_group,
        weights=[0.25, 0.75],
        t0=0.0,
        node_label="source",
    )
    assert len(source.lightcurves) == 2
    assert set(source.filters) == {"u", "g", "r"}
    assert source.apply_redshift is False

    # We fail if we try to turn on redshift.
    with pytest.raises(NotImplementedError):
        source.set_apply_redshift(True)

    # Check that we sample the lightcurves correctly.
    graph_state = source.sample_parameters(num_samples=1_000)
    lc_used = graph_state["source"]["selected_lightcurve"]
    assert np.all((lc_used == 0) | (lc_used == 1))
    assert np.sum(lc_used == 0) >= 200  # approximately 25% of the samples should be from lc1
    assert np.sum(lc_used == 1) >= 700  # approximately 75% of the samples should be from lc2

    # Check that the baseline values are set correctly for the selected lightcurves.
    # Lightcurve 0 has values given for u and g. Lightcurve 1 does not have any given
    # baseline values, so it should default to 0.0 in all bands.
    lc0_mask = lc_used == 0
    assert np.all(graph_state["source"]["baseline_u"][lc0_mask] == 0.1)
    assert np.all(graph_state["source"]["baseline_g"][lc0_mask] == 0.2)
    assert np.all(graph_state["source"]["baseline_r"][lc0_mask] == 0.0)  # Default

    lc1_mask = lc_used == 1
    assert np.all(graph_state["source"]["baseline_u"][lc1_mask] == 0.0)
    assert np.all(graph_state["source"]["baseline_g"][lc1_mask] == 0.0)
    assert np.all(graph_state["source"]["baseline_r"][lc1_mask] == 0.0)

    query_times = np.array([-1.0, 1.0, 2.0, 3.0, 4.0, 20.0, 21.0])
    query_filters = np.full(len(query_times), "g")
    fluxes = source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)
    assert len(fluxes) == 1_000

    # Check that we sampled from the lightcurve that we said we did.
    expected_0 = np.array([0.2, 3.0, 3.0, 3.0, 3.0, 0.2, 0.2])
    expected_1 = np.array([1.5, 1.5, 0.5, 1.5, 0.5, 0.5, 1.5])
    for idx in range(1_000):
        if lc_used[idx] == 0:
            # The first light curve is 3.0 when active
            assert np.allclose(fluxes[idx], expected_0)
        else:
            assert np.allclose(fluxes[idx], expected_1)

    # A call to get_band_fluxes with an unsupported filter should raise an error.
    query_filters[2] = "x"  # Change one filter to an unsupported one
    with pytest.raises(ValueError):
        source.get_band_fluxes(pb_group, query_times, query_filters, graph_state)


def test_create_multilightcurve_source_fail() -> None:
    """Test creating a MultiLightcurveSource with invalid parameters."""
    pb_group = _create_toy_passbands()

    # Lightcurve 1 is non-periodic and covers u and g.
    lc1_times = np.arange(0.0, 10.5, 0.5)
    lc1_lightcurves = {
        "u": np.array([lc1_times + 0.1, 2.0 * np.ones_like(lc1_times)]).T,
        "g": np.array([lc1_times, 3.0 * np.ones_like(lc1_times)]).T,
        "r": np.array([lc1_times, 3.0 * np.ones_like(lc1_times)]).T,
    }
    lc1_data = LightcurveData(lc1_lightcurves, lc_t0=0.0, baseline={"u": 0.1, "g": 0.2, "r": 0.3})

    # This single source works.
    _ = MultiLightcurveSource([lc1_data], pb_group, t0=0.0)

    # We fail with no t0 value.
    with pytest.raises(ValueError):
        _ = MultiLightcurveSource([lc1_data], pb_group)

    # Lightcurve 2 is periodic and covers r and g.
    lc2_times = np.arange(0.0, 19.0, 1.0)
    lc2_lightcurves = {
        "r": np.array([lc2_times, lc2_times % 2]).T,
        "g": np.array([lc2_times, lc2_times % 2 + 0.5]).T,
        "i": np.array([lc2_times, lc2_times % 2 + 0.5]).T,
    }
    lc2_data = LightcurveData(lc2_lightcurves, periodic=True)

    with pytest.raises(ValueError):
        # The passband group does not have data for the 'i' band.
        _ = MultiLightcurveSource([lc1_data, lc2_data], pb_group, weights=[0.25, 0.75], t0=0.0)


def test_create_multilightcurve_from_lclib_file(test_data_dir):
    """Test creating a MultiLightcurveSource from a LCLIB file."""
    passband_list = []
    pb_start = np.array([[400, 0.5], [500, 0.5], [600, 0.5]])
    pb_shift = np.array([[500, 0], [500, 0], [500, 0]])
    for idx, filter in enumerate(["u", "g", "r", "i", "z"]):
        pb = Passband(pb_start + idx * pb_shift, "LSST", filter)
        passband_list.append(pb)
    pb_group = PassbandGroup(given_passbands=passband_list)

    lc_file = test_data_dir / "test_lclib_data.TEXT"
    source = MultiLightcurveSource.from_lclib_file(lc_file, pb_group, t0=0.0)
    assert len(source.lightcurves) == 3
    assert set(source.filters) == {"u", "g", "r", "i", "z"}
    assert source.apply_redshift is False


def test_create_multilightcurve_from_lclib_file_filtered(test_data_dir):
    """Test creating a MultiLightcurveSource from a LCLIB file using a subset of filters."""
    passband_list = []
    pb_start = np.array([[400, 0.5], [500, 0.5], [600, 0.5]])
    pb_shift = np.array([[500, 0], [500, 0], [500, 0]])
    for idx, filter in enumerate(["u", "g", "r", "i", "z"]):
        pb = Passband(pb_start + idx * pb_shift, "LSST", filter)
        passband_list.append(pb)
    pb_group = PassbandGroup(given_passbands=passband_list)

    lc_file = test_data_dir / "test_lclib_data.TEXT"
    source = MultiLightcurveSource.from_lclib_file(lc_file, pb_group, t0=0.0, filters=["u", "g", "r"])
    assert len(source.lightcurves) == 3
    assert set(source.filters) == {"u", "g", "r"}
    assert source.apply_redshift is False
