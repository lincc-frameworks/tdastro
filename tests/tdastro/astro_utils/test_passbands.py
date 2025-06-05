import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sncosmo import Bandpass
from tdastro.astro_utils.passbands import Passband
from tdastro.sources.spline_model import SplineModel


def create_lsst_passband(path, filter_name, **kwargs):
    """Helper function to create an LSST Passband object for testing."""
    survey = "LSST"
    table_path = f"{path}/{survey}/{filter_name}.dat"
    return Passband.from_file(survey, filter_name, table_path=table_path, **kwargs)


def create_toy_passband(path, transmission_table, filter_name="a", **kwargs):
    """Helper function to create a toy Passband object for testing."""
    survey = "TOY"
    dir_path = Path(path, survey)
    dir_path.mkdir(parents=True, exist_ok=True)
    table_path = dir_path / f"{filter_name}.dat"

    # Create a transmission table file
    with open(table_path, "w") as f:
        f.write(transmission_table)

    # Create a Passband object
    return Passband.from_file(survey, filter_name, table_path=table_path, **kwargs)


def test_passband_str(passbands_dir, tmp_path):
    """Test the __str__ method of the Passband class."""
    # Test an LSST passband:
    LSST_u = create_lsst_passband(passbands_dir, "u")
    assert str(LSST_u) == "Passband: LSST_u"

    # Test a toy passband:
    transmission_table = "1000 0.5\n1005 0.6\n1010 0.7\n"
    a_band = create_toy_passband(tmp_path, transmission_table)
    assert str(a_band) == "Passband: TOY_a"


def test_passband_eq(passbands_dir, tmp_path):
    """Test the __str__ method of the Passband class."""
    a_band = Passband(np.array([[1000, 0.5], [1005, 0.6], [1010, 0.7]]), "LSST", "a")
    b_band = Passband(np.array([[1000, 0.5], [1005, 0.6], [1010, 0.7]]), "LSST", "b")
    c_band = Passband(np.array([[1000, 0.5], [1005, 0.7], [1010, 0.7]]), "LSST", "c")
    d_band = Passband(np.array([[1000, 0.5], [1005, 0.6], [1020, 0.7]]), "LSST", "d")
    assert a_band == b_band
    assert a_band != c_band
    assert a_band != d_band


def test_passband_manual_create(tmp_path):
    """Test that we can create a passband from the transmission table."""
    # Test we get a TypeError if we don't provide a survey and filter_name
    with pytest.raises(TypeError):
        _ = Passband()

    # Create a manual passband.
    transmission_table = np.array([[1000, 0.5], [1005, 0.6], [1010, 0.7]])
    test_pb = Passband(transmission_table, survey="test", filter_name="u")

    assert test_pb.survey == "test"
    assert test_pb.filter_name == "u"
    assert test_pb.full_name == "test_u"
    np.testing.assert_allclose(test_pb._loaded_table, transmission_table)

    # We can create an load a table in nm as well. It will auto-convert to Angstroms.
    transmission2 = np.array([[100, 0.5], [100.5, 0.6], [101, 0.7]])
    test_pb2 = Passband(transmission2, survey="test", filter_name="g", units="nm")
    assert test_pb2.survey == "test"
    assert test_pb2.filter_name == "g"
    assert test_pb2.full_name == "test_g"
    np.testing.assert_allclose(test_pb2._loaded_table, transmission_table)

    # We raise an error if the data is not sorted.
    with pytest.raises(ValueError):
        _ = Passband(
            np.array([[1000, 0.5], [900, 0.2], [1100, 0.7]]),
            survey="test",
            filter_name="u",
        )

    # We raise an error if the data is not the right shape.
    with pytest.raises(ValueError):
        _ = Passband(
            np.full((10, 3), 1.0),
            survey="test",
            filter_name="u",
        )

    # We raise an error with an invalid unit.
    with pytest.raises(ValueError):
        _ = Passband(
            transmission_table,
            survey="test",
            filter_name="u",
            units="invalid",
        )


def test_passband_load_transmission_table(passbands_dir, tmp_path):
    """Test the _load_transmission_table method of the Passband class."""
    # Test that we can load a standard LSST transmission table from the local test data
    LSST_g = create_lsst_passband(passbands_dir, "g")
    assert LSST_g._loaded_table is not None
    assert LSST_g._loaded_table.shape[1] == 2  # Two columns: wavelengths and transmissions
    assert LSST_g.waves is not None
    assert len(LSST_g.waves.shape) == 1

    # Test a toy transmission table was loaded correctly
    transmission_table = "1000 0.5\n1005 0.6\n1010 0.7\n"
    a_band = create_toy_passband(tmp_path, transmission_table)
    np.testing.assert_allclose(a_band._loaded_table, np.array([[1000, 0.5], [1005, 0.6], [1010, 0.7]]))

    # Test that we raise an error if the transmission table is blank
    transmission_table = ""
    test_pb_file_name = Path(tmp_path) / "r.dat"
    with open(test_pb_file_name, "w") as f:
        f.write(transmission_table)
    with warnings.catch_warnings():
        # Ignore the warning that we are loading an empty file.
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            a_band.load_transmission_table(test_pb_file_name)

    # Test that we raise an error if the transmission table is not formatted correctly
    transmission_table = "1000\n1005 0.6\n1010 0.7"
    with open(test_pb_file_name, "w") as f:
        f.write(transmission_table)
    with pytest.raises(ValueError):
        a_band.load_transmission_table(test_pb_file_name)

    # Test that we raise an error if the transmission table wavelengths are not sorted
    transmission_table = "1000 0.5\n900 0.6\n1010 0.7"
    with open(test_pb_file_name, "w") as f:
        f.write(transmission_table)
    with pytest.raises(ValueError):
        a_band.load_transmission_table(test_pb_file_name)


def test_passband_download_transmission_table(tmp_path):
    """Test the functionality of downloading the transmission table Passband class."""
    # Initialize a Passband object
    survey = "TEST"
    filter_name = "a"
    table_path = tmp_path / f"{survey}/{filter_name}.dat"
    table_url = f"https://data.lsdb.io/{survey}/{filter_name}.dat"
    transmission_table = "1000 0.5\n1005 0.6\n1010 0.7\n"

    def mock_urlretrieve(url, known_hash, fname, path):
        full_name = path / fname
        with open(full_name, "w") as f:
            f.write(transmission_table)
        return full_name

    # Mock the urlretrieve portion of the download method
    with patch("pooch.retrieve", side_effect=mock_urlretrieve) as mocked_urlretrieve:
        a_band = Passband.from_file(survey, filter_name, table_path=table_path, table_url=table_url)

        # Check that the transmission table was downloaded
        mocked_urlretrieve.assert_called_once_with(
            url=table_url,
            known_hash=None,
            fname=table_path.name,
            path=table_path.parent,
        )

        # Check that the transmission table was loaded correctly
        np.testing.assert_allclose(a_band._loaded_table, np.array([[1000, 0.5], [1005, 0.6], [1010, 0.7]]))


def test_passband_from_file(passbands_dir, tmp_path):
    """Test the from_file constructor of the Passband class."""
    # Test a toy transmission table was loaded correctly
    transmission_table = "1000 0.5\n1005 0.6\n1010 0.7\n"
    a_band = create_toy_passband(tmp_path, transmission_table)
    np.testing.assert_allclose(a_band._loaded_table, np.array([[1000, 0.5], [1005, 0.6], [1010, 0.7]]))

    table_path = Path(tmp_path, "TOY", "a.dat")
    a_band_2 = Passband.from_file("TOY", "a", table_path=table_path)
    np.testing.assert_allclose(a_band_2._loaded_table, np.array([[1000, 0.5], [1005, 0.6], [1010, 0.7]]))

    # We raise an error if the transmission table does not exist.
    with pytest.raises(ValueError):
        _ = Passband.from_file(survey="test", filter_name="u", table_path="./no_such_file.dat")


def test_passband_from_sncosmo(passbands_dir):
    """Test the from_sncosmo constructor of the Passband class."""
    sn_pb = Bandpass(
        np.array([6000, 6005, 6010]),  # wavelengths (A)
        np.array([0.5, 0.6, 0.7]),  # transmissions
    )
    ztf_band = Passband.from_sncosmo("ZTF", "g", sn_pb)
    assert ztf_band.survey == "ZTF"
    assert ztf_band.filter_name == "g"
    assert ztf_band.full_name == "ZTF_g"
    assert ztf_band._loaded_table is not None
    assert np.allclose(ztf_band._loaded_table[:, 0], sn_pb.wave)
    assert np.allclose(ztf_band._loaded_table[:, 1], sn_pb.trans)


def test_process_transmission_table(passbands_dir, tmp_path):
    """Test the process_transmission_table method of the Passband class; check correct methods are called."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_toy_passband(tmp_path, transmission_table)

    # Mock the methods that _process_transmission_table wraps, to check each is called once
    with (
        patch.object(a_band, "_interpolate_transmission_table") as mock_interp_table,
        patch.object(a_band, "_trim_transmission_by_quantile") as mock_trim_table,
        patch.object(a_band, "_normalize_transmission_table") as mock_norm_table,
    ):
        # Call the _process_transmission_table method
        delta_wave, trim_quantile = 1.0, 0.05
        a_band.process_transmission_table(delta_wave, trim_quantile)

        # Check that each method is called once
        mock_interp_table.assert_called_once_with(a_band._loaded_table, delta_wave)
        mock_trim_table.assert_called_once()
        mock_norm_table.assert_called_once()

    # Now call without mocking, to check waves set correctly (other values checked in method-specific tests)
    delta_wave, trim_quantile = 5.0, None
    a_band.process_transmission_table(delta_wave, trim_quantile)
    np.testing.assert_allclose(a_band.waves, np.arange(100, 301, delta_wave))

    # Check that we can call the method on a standard LSST transmission table
    LSST_r = create_lsst_passband(passbands_dir, "r")
    LSST_r.process_transmission_table(delta_wave, trim_quantile)
    assert LSST_r.waves is not None

    # Check that we raise an error if the transmission table is the wrong shape
    a_band._loaded_table = np.array([[100, 0.5, 105, 0.6]])
    with pytest.raises(ValueError):
        a_band.process_transmission_table(delta_wave, trim_quantile)


def test_interpolate_transmission_table(passbands_dir, tmp_path):
    """Test the _interpolate_transmission_table method of the Passband class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_toy_passband(tmp_path, transmission_table)

    # Interpolate the transmission table to a large step size (50 Angstrom)
    table = a_band._interpolate_transmission_table(a_band._loaded_table, delta_wave=50)
    np.testing.assert_allclose(table[:, 0], np.arange(100, 301, 50))

    # Interpolate the transmission table to a somewhat small step size (1 Angstrom)
    table = a_band._interpolate_transmission_table(a_band._loaded_table, delta_wave=1)
    np.testing.assert_allclose(table[:, 0], np.arange(100, 301, 1))

    # Interpolate the transmission table to an even smaller step size (0.1 Angstrom)
    table = a_band._interpolate_transmission_table(a_band._loaded_table, delta_wave=0.1)
    np.testing.assert_allclose(table[:, 0], np.arange(100, 300.01, 0.1))

    # Check that we raise an error if the transmission table is the wrong shape
    with pytest.raises(ValueError):
        a_band._interpolate_transmission_table(np.array([[100, 0.5], [200, 0.75], [300, 0.25], [400]]))

    # Check that we raise an error if the delta_wave is not a positive number
    with pytest.raises(ValueError):
        a_band._interpolate_transmission_table(a_band._loaded_table, delta_wave=-1)

    # Check that we can run method on a standard LSST transmission table
    LSST_i = create_lsst_passband(passbands_dir, "i")
    table = LSST_i._interpolate_transmission_table(LSST_i._loaded_table, delta_wave=50)
    assert len(table) > 0
    assert len(table) < len(LSST_i._loaded_table)


def test_trim_transmission_table(passbands_dir, tmp_path):
    """Test the _trim_transmission_by_quantile method of the Passband class."""

    # Test: transmission table with only 3 points
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_toy_passband(tmp_path, transmission_table)

    # Trim the transmission table by 5% (should remove the last point)
    table = a_band._trim_transmission_by_quantile(a_band._loaded_table, trim_quantile=0.05)
    np.testing.assert_allclose(table, np.array([[100, 0.5], [200, 0.75]]))

    # Trim the transmission table by 40% (should still preserve the first point, as table is skewed)
    table = a_band._trim_transmission_by_quantile(a_band._loaded_table, trim_quantile=0.4)
    np.testing.assert_allclose(table, np.array([[100, 0.5], [200, 0.75]]))

    # Check no trimming if quantile is 0
    table = a_band._trim_transmission_by_quantile(a_band._loaded_table, trim_quantile=None)
    np.testing.assert_allclose(table, np.array([[100, 0.5], [200, 0.75], [300, 0.25]]))

    # Check we raise an error if the quantile is not greater or equal to 0 and less than 0.5
    with pytest.raises(ValueError):
        a_band._trim_transmission_by_quantile(a_band._loaded_table, trim_quantile=0.5)
    with pytest.raises(ValueError):
        a_band._trim_transmission_by_quantile(a_band._loaded_table, trim_quantile=-0.1)

    # Check we raise an error if the transmission table is the wrong shape
    with pytest.raises(ValueError):
        a_band._trim_transmission_by_quantile(np.array([[100, 0.5], [200, 0.75], [300, 0.25], [400]]))

    # Test 2: larger, more gaussian transmission table
    transmission_table = "100 0.05\n200 0.1\n300 0.25\n400 0.5\n500 0.8\n600 0.6\n700 0.4\n800 0.2\n900 0.1\n"
    b_band = create_toy_passband(tmp_path, transmission_table, filter_name="b")

    # Trim the transmission table by 5% (should remove the last point)
    table = b_band._trim_transmission_by_quantile(b_band._loaded_table, trim_quantile=0.05)
    np.testing.assert_allclose(
        table,
        np.array(
            [[100, 0.05], [200, 0.1], [300, 0.25], [400, 0.5], [500, 0.8], [600, 0.6], [700, 0.4], [800, 0.2]]
        ),
    )

    # Trim the transmission table by 40% (should remove most points)
    table = b_band._trim_transmission_by_quantile(b_band._loaded_table, trim_quantile=0.4)
    np.testing.assert_allclose(table, np.array([[300, 0.25], [400, 0.5], [500, 0.8]]))

    # Check no trimming if quantile is 0
    table = b_band._trim_transmission_by_quantile(b_band._loaded_table, trim_quantile=None)
    np.testing.assert_allclose(
        table,
        np.array(
            [
                [100, 0.05],
                [200, 0.1],
                [300, 0.25],
                [400, 0.5],
                [500, 0.8],
                [600, 0.6],
                [700, 0.4],
                [800, 0.2],
                [900, 0.1],
            ]
        ),
    )

    # Test 3: much larger transmission table
    rng = np.random.default_rng(100)
    transmissions = rng.normal(0.5, 0.1, 1000)
    wavelengths = np.arange(100, 1100, 1)
    transmission_table = "\n".join(
        [
            f"{wavelength} {transmission}"
            for wavelength, transmission in zip(wavelengths, transmissions, strict=False)
        ]
    )
    c_band = create_toy_passband(tmp_path, transmission_table, filter_name="c")

    # Trim the transmission table by 5% on each side
    table = c_band._trim_transmission_by_quantile(c_band._loaded_table, trim_quantile=0.05)
    assert len(table) < len(c_band._loaded_table)

    original_area = np.trapezoid(c_band._loaded_table[:, 1], x=c_band._loaded_table[:, 0])
    trimmed_area = np.trapezoid(table[:, 1], x=table[:, 0])
    assert trimmed_area >= (original_area * 0.9)

    # Test 4: LSST transmission table
    LSST_z = create_lsst_passband(passbands_dir, "z")
    table = LSST_z._trim_transmission_by_quantile(LSST_z._loaded_table, trim_quantile=0.05)
    assert len(table) < len(LSST_z._loaded_table)

    original_area = np.trapezoid(LSST_z._loaded_table[:, 1], x=LSST_z._loaded_table[:, 0])
    trimmed_area = np.trapezoid(table[:, 1], x=table[:, 0])
    assert trimmed_area >= (original_area * 0.9)


def test_passband_normalize_transmission_table(passbands_dir, tmp_path):
    """Test the _normalize_transmission_table method of the Passband class."""
    # Test a toy transmission table
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_toy_passband(tmp_path, transmission_table, delta_wave=100, trim_quantile=None)

    # Normalize the transmission table (we skip interpolation as grid already matches)
    a_band._normalize_transmission_table(a_band._loaded_table)

    # Compare results
    expected_result = np.array([[100.0, 0.0075], [200.0, 0.005625], [300.0, 0.00125]])
    assert np.allclose(a_band.processed_transmission_table, expected_result)

    # Test that we raise an error if the transmission table is the wrong size or shape
    with pytest.raises(ValueError):
        a_band._normalize_transmission_table(np.array([[]]))
    with pytest.raises(ValueError):
        a_band._normalize_transmission_table(np.array([[100, 0.5]]))
    with pytest.raises(ValueError):
        a_band._normalize_transmission_table(np.array([100, 0.5]))
    with pytest.raises(ValueError):
        a_band._normalize_transmission_table(np.array([[100, 0.5, 105, 0.6]]))

    # Test we can call method on a standard LSST transmission table
    LSST_y = create_lsst_passband(passbands_dir, "y")
    normalized_table = LSST_y._normalize_transmission_table(LSST_y._loaded_table)
    assert len(normalized_table) == len(LSST_y._loaded_table)
    assert not np.allclose(normalized_table, LSST_y._loaded_table)


def test_passband_fluxes_to_bandflux(passbands_dir, tmp_path):
    """Test the fluxes_to_bandflux method of the Passband class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_toy_passband(tmp_path, transmission_table, delta_wave=100, trim_quantile=None)

    # Define some mock flux values and calculate our expected bandflux
    flux = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    expected_in_band_flux = np.trapezoid(flux * a_band.processed_transmission_table[:, 1], x=a_band.waves)
    in_band_flux = a_band.fluxes_to_bandflux(flux)
    np.testing.assert_allclose(in_band_flux, expected_in_band_flux)

    # Test with a different set of fluxes, regridding the transmission table
    a_band.process_transmission_table(delta_wave=50, trim_quantile=None)
    flux = np.array(
        [
            [100.0, 12.0, 1.0, 0.5, 0.25],
            [300.0, 200.0, 100.0, 50.0, 25.0],
            [1.0, 500, 1000, 500, 250],
            [2.0, 1.0, 1.0, 1.0, 1.0],
            [3.0, 1.0, 1.0, 1.0, 1.0],
            [200.0, 100.0, 50.0, 25.0, 12.5],
            [100.0, 50.0, 25.0, 12.5, 6.25],
        ]
    )
    in_band_flux = a_band.fluxes_to_bandflux(flux)
    expected_in_band_flux = np.trapezoid(
        flux * a_band.processed_transmission_table[:, 1], x=a_band.processed_transmission_table[:, 0]
    )
    np.testing.assert_allclose(in_band_flux, expected_in_band_flux)

    # Test we raise an error if the fluxes are not the right shape
    with pytest.raises(ValueError):
        a_band.fluxes_to_bandflux(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        a_band.fluxes_to_bandflux(np.full((7, 5, 2, 2), 1.0))

    # Test we raise an error if the fluxes are empty
    with pytest.raises(ValueError):
        a_band.fluxes_to_bandflux(np.array([]))

    # Test we can call method on a standard LSST transmission table
    LSST_u = create_lsst_passband(passbands_dir, "u")
    flux = np.random.rand(5, len(LSST_u.waves))
    in_band_flux = LSST_u.fluxes_to_bandflux(flux)
    assert in_band_flux is not None
    assert len(in_band_flux) == 5


def test_passband_fluxes_to_bandflux_mult_samples(passbands_dir, tmp_path):
    """Test the fluxes_to_bandflux method of the Passband class with multiple samples."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_toy_passband(tmp_path, transmission_table, delta_wave=100, trim_quantile=None)

    # Define some mock flux values and calculate our expected bandflux
    flux = np.array(
        [
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [3.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [
                [2.0, 2.0, 2.0],
                [4.0, 2.0, 2.0],
                [6.0, 2.0, 2.0],
                [4.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
            ],
        ]
    )
    expected = np.array([[1.0, 1.375, 1.75, 1.375, 1.0], [2.0, 2.75, 3.5, 2.75, 2.0]])

    result = a_band.fluxes_to_bandflux(flux)
    np.testing.assert_allclose(result, expected)


def test_passband_wrapped_from_physical_source(passbands_dir, tmp_path):
    """Test get_band_fluxes, PhysicalModel's wrapped version of Passband's fluxes_to_bandflux.."""
    # Set up physical model
    times = np.array([1.0, 2.0, 3.0])
    wavelengths = np.array([10.0, 20.0, 30.0])
    fluxes = np.array([[1.0, 5.0, 1.0], [5.0, 10.0, 5.0], [1.0, 5.0, 3.0]])
    model = SplineModel(times, wavelengths, fluxes, time_degree=1, wave_degree=1)
    state = model.sample_parameters()

    test_times = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    # Test with a single toy passband (see PassbandGroup tests for group tests)
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_toy_passband(tmp_path, transmission_table, delta_wave=100, trim_quantile=None)
    result_from_source_model = model.get_band_fluxes(a_band, test_times, filters=None, state=state)

    evaluated_fluxes = model.evaluate(test_times, a_band.waves, state)
    result_from_passband = a_band.fluxes_to_bandflux(evaluated_fluxes)
    np.testing.assert_allclose(result_from_source_model, result_from_passband)

    # Test with a standard LSST passband
    LSST_g = create_lsst_passband(passbands_dir, "g")
    result_from_source_model = model.get_band_fluxes(
        LSST_g, test_times, filters=np.repeat("g", len(test_times)), state=state
    )

    evaluated_fluxes = model.evaluate(test_times, LSST_g.waves, state)
    result_from_passband = LSST_g.fluxes_to_bandflux(evaluated_fluxes)
    np.testing.assert_allclose(result_from_source_model, result_from_passband)
