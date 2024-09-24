from unittest.mock import patch

import numpy as np
import pytest
from tdastro.astro_utils.passbands import Passband
from tdastro.sources.spline_model import SplineModel


def test_passband_init():
    """Test the initialization of the Passband class."""
    # Test we get a TypeError if we don't provide a survey and filter_name
    with pytest.raises(TypeError):
        a_band = Passband()

    # Test we get an NotImplementedError if we don't have a local transmission table or provided URL, and the
    # survey has no default URL
    with pytest.raises(NotImplementedError):
        a_band = Passband("TEST", "a")

    # Test that the Passband class can be initialized with a survey and filter_name
    # Mock a transmission table file that we'll "find" at passbands/TEST/a.dat
    transmission_table_array = np.array([[100, 0.5], [200, 0.75], [300, 0.25]])

    def mock_load_transmission_table(self, **kwargs):
        self._loaded_table = transmission_table_array

    with patch.object(Passband, "_load_transmission_table", new=mock_load_transmission_table):
        a_band = Passband("TEST", "a")
        assert a_band.survey == "TEST"
        assert a_band.filter_name == "a"
        assert a_band.full_name == "TEST_a"
        # Cannot check table_path as we have mocked the method that sets it; see unit test for loading tables
        assert a_band.table_url is None
        np.testing.assert_allclose(a_band._loaded_table, transmission_table_array)
        assert a_band.waves is not None


def create_test_passband(path, transmission_table, filter_name="a", **kwargs):
    """Helper function to create a Passband object for testing."""
    survey = "TEST"
    table_path = path / f"{survey}_{filter_name}.dat"

    # Create a transmission table file
    with open(table_path, "w") as f:
        f.write(transmission_table)

    # Create a Passband object
    return Passband(survey, filter_name, table_path=table_path, **kwargs)


def test_passband_str(tmp_path):
    """Test the __str__ method of the Passband class."""
    transmission_table = "1000 0.5\n1005 0.6\n1010 0.7\n"

    # Check that the __str__ method returns the expected string
    a_band = create_test_passband(tmp_path, transmission_table)
    assert str(a_band) == "Passband: TEST_a"


def test_passband_load_transmission_table(tmp_path):
    """Test the _load_transmission_table method of the Passband class."""

    # Test a toy transmission table was loaded correctly
    transmission_table = "1000 0.5\n1005 0.6\n1010 0.7\n"
    a_band = create_test_passband(tmp_path, transmission_table)
    np.testing.assert_allclose(a_band._loaded_table, np.array([[1000, 0.5], [1005, 0.6], [1010, 0.7]]))

    # Test that we raise an error if the transmission table is blank
    transmission_table = ""
    with open(a_band.table_path, "w") as f:
        f.write(transmission_table)
    with pytest.raises(ValueError):
        a_band._load_transmission_table()

    # Test that we raise an error if the transmission table is not formatted correctly
    transmission_table = "1000\n1005 0.6\n1010 0.7"
    with open(a_band.table_path, "w") as f:
        f.write(transmission_table)
    with pytest.raises(ValueError):
        a_band._load_transmission_table()

    # Test that we raise an error if the transmission table wavelengths are not sorted
    transmission_table = "1000 0.5\n900 0.6\n1010 0.7"
    with open(a_band.table_path, "w") as f:
        f.write(transmission_table)
    with pytest.raises(ValueError):
        a_band._load_transmission_table()


def test_passband_download_transmission_table(tmp_path):
    """Test the _download_transmission_table method of the Passband class."""
    # Initialize a Passband object
    survey = "TEST"
    filter_name = "a"
    table_path = tmp_path / f"{survey}_{filter_name}.dat"
    table_url = f"http://example.com/{survey}/{filter_name}.dat"
    transmission_table = "1000 0.5\n1005 0.6\n1010 0.7\n"

    def mock_urlretrieve(url, filename, *args, **kwargs):  # Urlretrieve saves contents directly to filename
        with open(filename, "w") as f:
            f.write(transmission_table)
        return filename, None

    # Mock the urlretrieve portion of the download method
    with patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve) as mocked_urlretrieve:
        a_band = Passband(survey, filter_name, table_path=table_path, table_url=table_url)

        # Check that the transmission table was downloaded
        mocked_urlretrieve.assert_called_once_with(table_url, table_path)

        # Check that the transmission table was loaded correctly
        np.testing.assert_allclose(a_band._loaded_table, np.array([[1000, 0.5], [1005, 0.6], [1010, 0.7]]))


def test_process_transmission_table(tmp_path):
    """Test the process_transmission_table method of the Passband class; check correct methods are called."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_test_passband(tmp_path, transmission_table)

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


def test_interpolate_transmission_table(tmp_path):
    """Test the _interpolate_transmission_table method of the Passband class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_test_passband(tmp_path, transmission_table)

    # Interpolate the transmission table to a large step size (50 Angstrom)
    table = a_band._interpolate_transmission_table(a_band._loaded_table, delta_wave=50)
    np.testing.assert_allclose(table[:, 0], np.arange(100, 301, 50))

    # Interpolate the transmission table to a somewhat small step size (1 Angstrom)
    table = a_band._interpolate_transmission_table(a_band._loaded_table, delta_wave=1)
    np.testing.assert_allclose(table[:, 0], np.arange(100, 301, 1))

    # Interpolate the transmission table to an even smaller step size (0.1 Angstrom)
    table = a_band._interpolate_transmission_table(a_band._loaded_table, delta_wave=0.1)
    np.testing.assert_allclose(table[:, 0], np.arange(100, 300.01, 0.1))


def test_trim_transmission_table(tmp_path):
    """Test the _trim_transmission_by_quantile method of the Passband class."""

    # Test: transmission table with only 3 points
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_test_passband(tmp_path, transmission_table)

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

    # Test 2: larger, more normal transmission table
    transmission_table = "100 0.05\n200 0.1\n300 0.25\n400 0.5\n500 0.8\n600 0.6\n700 0.4\n800 0.2\n900 0.1\n"
    b_band = create_test_passband(tmp_path, transmission_table, filter_name="b")

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
    transmissions = np.random.normal(0.5, 0.1, 1000)
    wavelengths = np.arange(100, 1100, 1)
    transmission_table = "\n".join(
        [f"{wavelength} {transmission}" for wavelength, transmission in zip(wavelengths, transmissions)]
    )
    c_band = create_test_passband(tmp_path, transmission_table, filter_name="c")

    # Trim the transmission table by 5% on each side
    table = c_band._trim_transmission_by_quantile(c_band._loaded_table, trim_quantile=0.05)
    assert len(table) < len(c_band._loaded_table)

    original_area = np.trapz(c_band._loaded_table[:, 1], x=c_band._loaded_table[:, 0])
    trimmed_area = np.trapz(table[:, 1], x=table[:, 0])
    assert trimmed_area >= (original_area * 0.9)


def test_passband_normalize_transmission_table(tmp_path):
    """Test the _normalize_transmission_table method of the Passband class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_test_passband(tmp_path, transmission_table, delta_wave=100, trim_quantile=None)

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


def test_passband_fluxes_to_bandflux(tmp_path):
    """Test the fluxes_to_bandflux method of the Passband class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_test_passband(tmp_path, transmission_table, delta_wave=100, trim_quantile=None)

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
    expected_in_band_flux = np.trapz(flux * a_band.processed_transmission_table[:, 1], x=a_band.waves)
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
    expected_in_band_flux = np.trapz(
        flux * a_band.processed_transmission_table[:, 1], x=a_band.processed_transmission_table[:, 0]
    )
    np.testing.assert_allclose(in_band_flux, expected_in_band_flux)

    # Test we raise an error if the fluxes are not the right shape
    with pytest.raises(ValueError):
        a_band.fluxes_to_bandflux(np.array([[1.0, 2.0], [3.0, 4.0]]))

    # Test we raise an error if the fluxes are empty
    with pytest.raises(ValueError):
        a_band.fluxes_to_bandflux(np.array([]))


def test_passband_wrapped_from_physical_source(tmp_path):
    """Test get_band_fluxes, PhysicalModel's wrapped version of Passband's fluxes_to_bandflux.."""
    times = np.array([1.0, 2.0, 3.0])
    wavelengths = np.array([10.0, 20.0, 30.0])
    fluxes = np.array([[1.0, 5.0, 1.0], [5.0, 10.0, 5.0], [1.0, 5.0, 3.0]])
    model = SplineModel(times, wavelengths, fluxes, time_degree=1, wave_degree=1)
    state = model.sample_parameters()

    test_times = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    # Try with a single passband (see PassbandGroup tests for group tests)
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_test_passband(tmp_path, transmission_table, delta_wave=100, trim_quantile=None)
    result_from_source_model = model.get_band_fluxes(a_band, test_times, state)

    evaluated_fluxes = model.evaluate(test_times, a_band.waves, state)
    result_from_passband = a_band.fluxes_to_bandflux(evaluated_fluxes)

    np.testing.assert_allclose(result_from_source_model, result_from_passband)
