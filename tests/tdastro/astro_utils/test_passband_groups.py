import numpy as np
import pytest
from tdastro.astro_utils.passbands import Passband, PassbandGroup
from tdastro.sources.spline_model import SplineModel


def create_lsst_passband_group(passbands_dir, delta_wave=5.0, trim_quantile=None):
    """Helper function to create a PassbandGroup object for LSST passbands, using transmission tables
    included in the test data directory."""
    return PassbandGroup(
        preset="LSST", table_dir=passbands_dir, delta_wave=delta_wave, trim_quantile=trim_quantile
    )


def create_toy_passband_group(path, delta_wave=5.0, trim_quantile=None):
    """Helper function to create a toy PassbandGroup object for testing.

    Notable features of this toy passband group:
    - Three passbands: A, B, C
    - Each transmission table includes only three rows
    - Passbands A and B have overlapping wavelength ranges
    - Passbands A and C are defined with a 100 unit grid step, while passband B is defined with a 50 unit step
    """
    # Initialize requirements for PassbandGroup object
    survey = "TOY"
    filter_names = ["a", "b", "c"]
    table_dir = path / survey
    table_dir.mkdir()

    transmission_tables = {
        "a": "100 0.5\n200 0.75\n300 0.25\n",
        "b": "250 0.25\n300 0.5\n350 0.75\n",
        "c": "400 0.75\n500 0.25\n600 0.5\n",
    }
    for filter_name in filter_names:
        with open(table_dir / f"{filter_name}.dat", "w") as f:
            f.write(transmission_tables[filter_name])

    # Load the PassbandGroup object
    passbands = []
    for filter_name in filter_names:
        passbands.append(
            {
                "survey": survey,
                "filter_name": filter_name,
                "table_path": table_dir / f"{filter_name}.dat",
                "delta_wave": delta_wave,
                "trim_quantile": trim_quantile,
            }
        )
    return PassbandGroup(passband_parameters=passbands)


def test_passband_group_init(tmp_path, passbands_dir):
    """Test the initialization of the Passband class, and implicitly, _load_preset."""
    # Test that we cannot create an empty PassbandGroup object
    with pytest.raises(ValueError):
        _ = PassbandGroup()

    # Test that the PassbandGroup class raises an error for an unknown preset
    try:
        _ = PassbandGroup(preset="Unknown")
    except ValueError as e:
        assert str(e) == "Unknown passband preset: Unknown"
    else:
        raise AssertionError("PassbandGroup should raise an error for an unknown preset")

    # Test that the PassbandGroup class can be initialized with a preset
    lsst_passband_group = create_lsst_passband_group(passbands_dir)
    assert len(lsst_passband_group.passbands) == 6
    assert "LSST_u" in lsst_passband_group.passbands
    assert "LSST_g" in lsst_passband_group.passbands
    assert "LSST_r" in lsst_passband_group.passbands
    assert "LSST_i" in lsst_passband_group.passbands
    assert "LSST_z" in lsst_passband_group.passbands
    assert "LSST_y" in lsst_passband_group.passbands

    # Test that the PassbandGroup class can be initialized with a dict of passband parameters
    lsst_gri_passband_parameters = [
        {"survey": "LSST", "filter_name": "g", "table_path": f"{passbands_dir}/LSST/g.dat"},
        {"survey": "LSST", "filter_name": "r", "table_path": f"{passbands_dir}/LSST/r.dat"},
        {"survey": "LSST", "filter_name": "i", "table_path": f"{passbands_dir}/LSST/i.dat"},
    ]
    lsst_gri_passband_group = PassbandGroup(passband_parameters=lsst_gri_passband_parameters)
    assert len(lsst_gri_passband_group.passbands) == 3
    assert "LSST_g" in lsst_gri_passband_group.passbands
    assert "LSST_r" in lsst_gri_passband_group.passbands
    assert "LSST_i" in lsst_gri_passband_group.passbands

    # Test our toy passband group, which makes a PassbandGroup using a custom passband parameters dictionary
    toy_passband_group = create_toy_passband_group(tmp_path)
    assert len(toy_passband_group.passbands) == 3
    assert "TOY_a" in toy_passband_group.passbands
    assert "TOY_b" in toy_passband_group.passbands
    assert "TOY_c" in toy_passband_group.passbands
    np.testing.assert_allclose(
        toy_passband_group.waves,
        np.unique(np.concatenate([np.arange(100, 301, 5), np.arange(250, 351, 5), np.arange(400, 601, 5)])),
    )

    # Test that the PassbandGroup class raises an error for an unknown preset
    try:
        _ = PassbandGroup(preset="Unknown")
    except ValueError as e:
        assert str(e) == "Unknown passband preset: Unknown"
    else:
        raise AssertionError("PassbandGroup should raise an error for an unknown preset")


def test_passband_group_from_list(tmp_path):
    """Test that we can create a PassbandGroup from a pre-specified list."""
    pb_list = [
        Passband(
            "my_survey",
            "a",
            table_values=np.array([[100, 0.5], [200, 0.75], [300, 0.25]]),
            trim_quantile=None,
        ),
        Passband(
            "my_survey",
            "b",
            table_values=np.array([[250, 0.25], [300, 0.5], [350, 0.75]]),
            trim_quantile=None,
        ),
        Passband(
            "my_survey",
            "c",
            table_values=np.array([[400, 0.75], [500, 0.25], [600, 0.5]]),
            trim_quantile=None,
        ),
    ]
    test_passband_group = PassbandGroup(given_passbands=pb_list)
    assert len(test_passband_group.passbands) == 3
    assert "my_survey_a" in test_passband_group.passbands
    assert "my_survey_b" in test_passband_group.passbands
    assert "my_survey_c" in test_passband_group.passbands

    assert np.allclose(
        test_passband_group.waves,
        np.unique(np.concatenate([np.arange(100, 301, 5), np.arange(250, 351, 5), np.arange(400, 601, 5)])),
    )


def test_passband_unique_waves():
    """Test that if we create two passbands with very similar wavelengths, they get merged."""
    pb_list = [
        Passband(
            "my_survey",
            "a",
            table_values=np.array([[100, 0.5], [250, 0.25]]),
            trim_quantile=None,
        ),
        Passband(
            "my_survey",
            "b",
            table_values=np.array([[200.000001, 0.25], [300.000001, 0.5]]),
            trim_quantile=None,
        ),
    ]
    test_passband_group = PassbandGroup(given_passbands=pb_list)
    assert np.allclose(
        test_passband_group.waves,
        np.concatenate([np.arange(100, 201, 5), np.arange(205.000001, 301, 5)]),
    )

    # Larger gaps won't register.
    pb_list = [
        Passband(
            "my_survey",
            "a",
            table_values=np.array([[100, 0.5], [250, 0.25]]),
            trim_quantile=None,
        ),
        Passband(
            "my_survey",
            "b",
            table_values=np.array([[200.5, 0.25], [300.5, 0.5]]),
            trim_quantile=None,
        ),
    ]
    test_passband_group = PassbandGroup(given_passbands=pb_list)
    assert np.allclose(
        test_passband_group.waves,
        np.unique(np.concatenate([np.arange(100, 251, 5), np.arange(200.5, 301, 5)])),
    )


def test_passband_group_str(passbands_dir, tmp_path):
    """Test the __str__ method of the PassbandGroup class."""
    toy_passband_group = create_toy_passband_group(tmp_path)
    assert str(toy_passband_group) == "PassbandGroup containing 3 passbands: TOY_a, TOY_b, TOY_c"

    lsst_passband_group = create_lsst_passband_group(passbands_dir)
    assert str(lsst_passband_group) == (
        "PassbandGroup containing 6 passbands: LSST_u, LSST_g, LSST_r, LSST_i, LSST_z, LSST_y"
    )


def test_passband_group_fluxes_to_bandfluxes(passbands_dir):
    """Test the fluxes_to_bandfluxes method of the PassbandGroup class."""
    # Test with simple passband group
    lsst_passband_group = create_lsst_passband_group(passbands_dir, delta_wave=20, trim_quantile=None)

    # Create some mock flux data with the same number of columns as wavelengths in our PassbandGroup
    flux = np.linspace(lsst_passband_group.waves * 10, 100, 5)
    bandfluxes = lsst_passband_group.fluxes_to_bandfluxes(flux)

    # Check structure of result is as expected (see Passband tests for value tests)
    assert len(bandfluxes) == 6
    for band_name in lsst_passband_group.passbands:
        assert band_name in bandfluxes
        assert bandfluxes[band_name].shape == (5,)


def test_passband_group_wrapped_from_physical_source(passbands_dir, tmp_path):
    """Test get_band_fluxes, PhysicalModel's wrapped version of PassbandGroup's fluxes_to_bandfluxes."""
    times = np.array([1.0, 2.0, 3.0])
    wavelengths = np.array([10.0, 20.0, 30.0])
    fluxes = np.array([[1.0, 5.0, 1.0], [5.0, 10.0, 5.0], [1.0, 5.0, 3.0]])
    model = SplineModel(times, wavelengths, fluxes, time_degree=1, wave_degree=1)
    state = model.sample_parameters()

    test_times = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    # Test wrapper with a PassbandGroup as input (see Passband tests for single-band tests)
    # Using LSST passband group:
    lsst_passband_group = create_lsst_passband_group(passbands_dir, delta_wave=20, trim_quantile=None)
    n_lsst_bands = len(lsst_passband_group.passbands)
    n_times = len(test_times)

    fluxes_source_model = model.get_band_fluxes(
        lsst_passband_group,
        times=np.repeat(test_times, n_lsst_bands),
        filters=np.tile(list(lsst_passband_group.passbands.keys()), n_times),
        state=state,
    )
    result_from_source_model = {
        filter_name: fluxes_source_model[i::n_lsst_bands]
        for i, filter_name in enumerate(lsst_passband_group.passbands)
    }
    evaluated_fluxes = model.evaluate(test_times, lsst_passband_group.waves, state)
    result_from_passband_group = lsst_passband_group.fluxes_to_bandfluxes(evaluated_fluxes)

    # Check the two dicts are the same
    assert result_from_source_model.keys() == result_from_passband_group.keys()
    for key in result_from_source_model:
        np.testing.assert_allclose(result_from_source_model[key], result_from_passband_group[key])

    # Using toy passband group:
    toy_passband_group = create_toy_passband_group(tmp_path, delta_wave=20, trim_quantile=None)
    n_toy_bands = len(toy_passband_group.passbands)
    fluxes_source_model = model.get_band_fluxes(
        toy_passband_group,
        times=np.repeat(test_times, n_toy_bands),
        filters=np.tile(list(toy_passband_group.passbands.keys()), n_times),
        state=state,
    )
    result_from_source_model = {
        filter_name: fluxes_source_model[i::n_toy_bands]
        for i, filter_name in enumerate(toy_passband_group.passbands)
    }
    evaluated_fluxes = model.evaluate(test_times, toy_passband_group.waves, state)
    result_from_passband_group = toy_passband_group.fluxes_to_bandfluxes(evaluated_fluxes)

    # Check the two dicts are the same
    assert result_from_source_model.keys() == result_from_passband_group.keys()
    for key in result_from_source_model:
        np.testing.assert_allclose(result_from_source_model[key], result_from_passband_group[key])


def test_passband_group_calculate_in_band_wave_indices(passbands_dir, tmp_path):
    """Test the calculate_in_band_wave_indices method of the PassbandGroup class using both toy and LSST
    transmission tables."""
    # Using LSST passband group:
    lsst_passband_group = create_lsst_passband_group(passbands_dir, delta_wave=20, trim_quantile=None)

    # Make sure group.waves contains the union of all passband waves with no duplicates or other values
    np.testing.assert_allclose(
        lsst_passband_group.waves,
        np.unique(np.concatenate([passband.waves for passband in lsst_passband_group.passbands.values()])),
    )

    # Using toy passband group:
    toy_passband_group = create_toy_passband_group(tmp_path, delta_wave=20, trim_quantile=None)

    passband_A = toy_passband_group.passbands["TOY_a"]
    passband_B = toy_passband_group.passbands["TOY_b"]
    passband_C = toy_passband_group.passbands["TOY_c"]

    # Note that passband_A and passband_B have overlapping wavelength ranges
    # Where passband_A covers 100-300 and passband_B covers 250-350 (and passband_C covers 400-600)
    np.testing.assert_allclose(
        passband_A._in_band_wave_indices, np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13])
    )
    np.testing.assert_allclose(passband_A.waves, toy_passband_group.waves[passband_A._in_band_wave_indices])

    np.testing.assert_allclose(passband_B._in_band_wave_indices, np.array([8, 10, 12, 14, 15, 16]))
    np.testing.assert_allclose(passband_B.waves, toy_passband_group.waves[passband_B._in_band_wave_indices])

    assert passband_C._in_band_wave_indices == slice(17, 28)
    np.testing.assert_allclose(
        passband_C.waves,
        toy_passband_group.waves[passband_C._in_band_wave_indices],
    )
