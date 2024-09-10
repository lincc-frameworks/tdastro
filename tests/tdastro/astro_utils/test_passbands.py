from unittest.mock import patch

import numpy as np
import pytest
from tdastro.astro_utils.passbands import Passband


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

    def mock_load_transmission_table(self):
        self._loaded_table = transmission_table_array

    with patch.object(Passband, "_load_transmission_table", new=mock_load_transmission_table):
        a_band = Passband("TEST", "a")
        assert a_band.survey == "TEST"
        assert a_band.filter_name == "a"
        assert a_band.full_name == "TEST_a"
        # Cannot check table_path as we have mocked the method that sets it; see unit test for loading tables
        assert a_band.table_url is None
        np.testing.assert_allclose(a_band._loaded_table, transmission_table_array)
        assert isinstance(a_band._wave_grid, np.ndarray)
        assert a_band.processed_transmission_table is not None


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


def test_passband_set_wave_grid_attr(tmp_path):
    """Test the _set_wave_grid_attr method of the Passband class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_test_passband(tmp_path, transmission_table, wave_grid=100)

    # Test that the wave grid is set correctly
    a_band._set_wave_grid_attr(50.0)
    np.testing.assert_allclose(a_band._wave_grid, np.arange(100.0, 301.0, 50.0))

    # Test that the wave grid is reset correctly
    a_band._set_wave_grid_attr(100.0)
    np.testing.assert_allclose(a_band._wave_grid, np.array([100.0, 200.0, 300.0]))

    # Test that we can set the wave grid with None
    a_band._set_wave_grid_attr(None)
    assert a_band._wave_grid is None

    # Test we can be lazy and set the wave grid with just an integer
    a_band._set_wave_grid_attr(100)
    np.testing.assert_allclose(a_band._wave_grid, np.array([100.0, 200.0, 300.0]))

    # Test that we can set the wave grid with a numpy array
    # Note, we truncate given grid to fit transmission table bounds
    a_band._set_wave_grid_attr(np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]))
    np.testing.assert_allclose(a_band._wave_grid, np.array([100.0, 150.0, 200.0, 250.0, 300.0]))

    # Test that we raise an error if the wave grid is not sorted
    with pytest.raises(ValueError):
        a_band._set_wave_grid_attr(np.array([100.0, 300.0, 200.0]))

    # Test that we raise an error if the wave grid is not 1D
    with pytest.raises(ValueError):
        a_band._set_wave_grid_attr(np.array([[100.0, 200.0], [300.0, 400.0]]))

    # Test we raise an error if wave grid is only one value
    with pytest.raises(ValueError):
        a_band._set_wave_grid_attr(np.array([100.0]))

    # Test we raise an error if wave grid is empty
    with pytest.raises(ValueError):
        a_band._set_wave_grid_attr(np.array([]))


def test_passband_set_transmission_table_grid(tmp_path):
    """Test the set_transmission_table_to_new_grid method of the PassbandGroup class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n400 0.5\n500 0.25\n600 0.5"

    # Test case where interpolation is not needed
    a_band = create_test_passband(tmp_path, transmission_table, wave_grid=100)
    np.testing.assert_allclose(a_band._wave_grid, np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]))
    np.testing.assert_allclose(
        a_band.processed_transmission_table[:, 0],
        np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
    )

    # Test that grid is reset successfully AND we have interpolated the transmission table as well
    a_band.set_transmission_table_grid(50.0)
    np.testing.assert_allclose(a_band._wave_grid, np.arange(100.0, 601.0, 50.0))
    np.testing.assert_allclose(
        a_band.processed_transmission_table[:, 0],
        np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0]),
    )


def test_passband_process_transmission_table(tmp_path):
    """Test the _process_transmission_table method of the Passband class: correct methods are called."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_test_passband(tmp_path, transmission_table, wave_grid=100)

    # Mock the methods that _process_transmission_table wraps
    with (
        patch.object(a_band, "_interpolate_or_downsample_transmission_table") as mock_interp_table,
        patch.object(a_band, "_normalize_interpolated_transmission_table") as mock_norm_table,
    ):
        # Call the _process_transmission_table method
        a_band._process_transmission_table()

        # Check that each method is called once
        mock_interp_table.assert_called_once_with(a_band._loaded_table)
        mock_norm_table.assert_called_once()


def test_passband_interpolate_or_downsample_transmission_table(tmp_path):
    """Test down-sampling and interpolation of the transmission table."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_test_passband(tmp_path, transmission_table)

    # Downsample the transmission table
    a_band._set_wave_grid_attr(200)
    table = a_band._interpolate_or_downsample_transmission_table(a_band._loaded_table)
    np.testing.assert_allclose(table[:, 0], [100, 300])

    # Downsample the transmission table to a very large step size
    a_band._set_wave_grid_attr(500)
    table = a_band._interpolate_or_downsample_transmission_table(a_band._loaded_table)
    np.testing.assert_allclose(table[:, 0], [100])

    # Interpolate the transmission table (note for comparison below: np.arange is exclusive of stop value)
    a_band._set_wave_grid_attr(50)
    table = a_band._interpolate_or_downsample_transmission_table(a_band._loaded_table)
    np.testing.assert_allclose(table[:, 0], np.arange(100, 301, 50))

    # Interpolate the transmission table to a somewhat small step size (1 Angstrom)
    a_band._set_wave_grid_attr(1)
    table = a_band._interpolate_or_downsample_transmission_table(a_band._loaded_table)
    np.testing.assert_allclose(table[:, 0], np.arange(100, 301, 1))

    # Interpolate the transmission table to an even smaller step size (0.1 Angstrom)
    a_band._set_wave_grid_attr(0.1)
    table = a_band._interpolate_or_downsample_transmission_table(a_band._loaded_table)
    np.testing.assert_allclose(table[:, 0], np.arange(100, 300.01, 0.1))


def test_passband_normalize_interpolated_transmission_table(tmp_path):
    """Test the _normalize_interpolated_transmission_table method of the Passband class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_test_passband(tmp_path, transmission_table, wave_grid=100)

    # Normalize the transmission table (we skip interpolation as grid already matches)
    a_band._normalize_interpolated_transmission_table(a_band._loaded_table)

    # Compare results
    expected_result = np.array([[100.0, 0.0075], [200.0, 0.005625], [300.0, 0.00125]])
    assert np.allclose(a_band.processed_transmission_table, expected_result)

    # Test that we raise an error if the transmission table is the wrong size or shape
    with pytest.raises(ValueError):
        a_band._normalize_interpolated_transmission_table(np.array([[]]))
    with pytest.raises(ValueError):
        a_band._normalize_interpolated_transmission_table(np.array([[100, 0.5]]))
    with pytest.raises(ValueError):
        a_band._normalize_interpolated_transmission_table(np.array([100, 0.5]))
    with pytest.raises(ValueError):
        a_band._normalize_interpolated_transmission_table(np.array([[100, 0.5, 105, 0.6]]))


def test_passband_interpolate_flux_densities_basic(tmp_path):
    """Test basic functionality of the _interpolate_flux_densities method of the Passband class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n400 0.5\n500 0.25\n600 0.5"

    # Test: no interpolation needed
    a_band = create_test_passband(tmp_path, transmission_table, wave_grid=100)
    fluxes = np.array([[0.25, 0.5, 1.0, 1.0, 0.5, 0.25]])
    flux_wavelengths = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])

    with patch("scipy.interpolate.InterpolatedUnivariateSpline") as mock_spline:
        (result_fluxes, result_wavelengths) = a_band._interpolate_flux_densities(fluxes, flux_wavelengths)
        np.testing.assert_allclose(result_fluxes, fluxes)
        np.testing.assert_allclose(result_wavelengths, flux_wavelengths)
        mock_spline.assert_not_called()

    # Test: interpolation when fluxes span entire band
    b_band = create_test_passband(tmp_path, transmission_table, wave_grid=50, filter_name="b")
    fluxes = np.array([[0.25, 0.5, 1.0, 1.0, 0.5, 0.25], [0.25, 0.5, 1.0, 1.0, 0.5, 0.25]])
    flux_wavelengths = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    (result_fluxes, result_wavelengths) = b_band._interpolate_flux_densities(fluxes, flux_wavelengths)

    # Check wavelengths
    expected_waves = np.arange(100, 601, 50)
    np.testing.assert_allclose(result_wavelengths, expected_waves)

    # Check the interpolated fluxes, while being somewhat agnostic to the specific method of interpolation
    assert result_fluxes.shape == (2, 11)
    # Ends should be 0.25
    assert result_fluxes[0, 0] == 0.25
    assert result_fluxes[0, 10] == 0.25
    # Beginning should be increasing. We don't go all the way to the middle, as the spline may wiggle a bit
    # approximating the flat top between 300 and 400 Angstroms.
    assert np.all(np.diff(result_fluxes[0, :5]) > 0)
    # End should be decreasing. Similarly, not starting exactly at the middle.
    assert np.all(np.diff(result_fluxes[0, 7:]) < 0)


def test_passband_interpolate_flux_densities_spline_degrees(tmp_path):
    """Test spline degree settings in the _interpolate_flux_densities method of the Passband class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n400 0.5\n500 0.25\n600 0.5"
    a_band = create_test_passband(tmp_path, transmission_table, wave_grid=50)
    fluxes = np.array([[0.25, 0.5, 1.0, 1.0, 0.5, 0.25], [0.25, 0.5, 1.0, 1.0, 0.5, 0.25]])
    flux_wavelengths = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])

    # Test we can run interpolation with spline_degree at default (3)
    (result_fluxes, result_wavelengths) = a_band._interpolate_flux_densities(fluxes, flux_wavelengths)
    assert result_fluxes.shape == (2, 11)
    assert result_wavelengths.shape == (11,)

    # Test we can run with spline_degree at values 1-5
    for degree in range(1, 6):
        (result_fluxes, result_wavelengths) = a_band._interpolate_flux_densities(
            fluxes, flux_wavelengths, spline_degree=degree
        )
        assert result_fluxes.shape == (2, 11)
        assert result_wavelengths.shape == (11,)

    # Test we raise an error for invalid degrees
    with pytest.raises(ValueError):
        a_band._interpolate_flux_densities(fluxes, flux_wavelengths, spline_degree=0)

    with pytest.raises(ValueError):
        a_band._interpolate_flux_densities(fluxes, flux_wavelengths, spline_degree=6)

    # Test we raise an error if number of degrees is greater than number of points
    fluxes = np.array([[0.25, 0.5]])
    flux_wavelengths = np.array([100.0, 200.0])
    with pytest.raises(ValueError):
        a_band._interpolate_flux_densities(fluxes, flux_wavelengths, spline_degree=3)


def test_passband_interpolate_flux_densities_truncation_needed(tmp_path):
    """Test truncation in the _interpolate_flux_densities method of the Passband class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n400 0.5\n500 0.25\n600 0.5"

    # Test truncation when fluxes are out of bounds
    a_band = create_test_passband(tmp_path, transmission_table, wave_grid=100)
    fluxes = np.array([[0.25, 0.5, 1.0, 1.0, 0.5, 0.25, 0.0, 0.5]])
    flux_wavelengths = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 650.0])
    (result_fluxes, result_wavelengths) = a_band._interpolate_flux_densities(fluxes, flux_wavelengths)
    assert result_fluxes.shape == (1, 6)
    assert result_wavelengths.shape == (6,)
    np.testing.assert_allclose(result_wavelengths, np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]))
    np.testing.assert_allclose(result_fluxes, fluxes[:, 1:7])

    # Test truncation and interpolation when fluxes are out of bounds
    a_band.set_transmission_table_grid(50)
    (result_fluxes, result_wavelengths) = a_band._interpolate_flux_densities(fluxes, flux_wavelengths)
    assert result_fluxes.shape == (1, 11)
    assert result_wavelengths.shape == (11,)
    np.testing.assert_allclose(result_wavelengths, np.arange(100, 601, 50))
    np.testing.assert_allclose(result_fluxes[:, ::2], fluxes[:, 1:7])


def test_passband_interpolate_flux_densities_padding_needed(tmp_path):
    """Test padding in the _interpolate_flux_densities method of the Passband class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n400 0.5\n500 0.25\n600 0.5"

    # Test padding (with NO interpolation) when fluxes do not span the entire band
    a_band = create_test_passband(tmp_path, transmission_table, wave_grid=100, filter_name="b")
    fluxes = np.array([[0.5, 1.0, 0.5, 0.25], [0.5, 1.0, 0.5, 0.25]])
    flux_wavelengths = np.array([200.0, 300.0, 400.0, 500.0])

    # Case 1: padding needed with extrapolation="raise"
    with pytest.raises(ValueError):
        a_band._interpolate_flux_densities(fluxes, flux_wavelengths, "raise")

    # Case 2: padding needed with extrapolation="zeros"
    (result_fluxes, result_wavelengths) = a_band._interpolate_flux_densities(
        fluxes, flux_wavelengths, "zeros"
    )
    expected_fluxes = np.array([[0.0, 0.5, 1.0, 0.5, 0.25, 0.0], [0.0, 0.5, 1.0, 0.5, 0.25, 0.0]])
    expected_waves = np.array([100, 200, 300, 400, 500, 600])
    np.testing.assert_allclose(result_fluxes, expected_fluxes)
    np.testing.assert_allclose(result_wavelengths, expected_waves)

    # Case 3: padding needed with extrapolation="constant"
    (result_fluxes, result_wavelengths) = a_band._interpolate_flux_densities(
        fluxes, flux_wavelengths, "const"
    )
    expected_fluxes = np.array([[0.5, 0.5, 1.0, 0.5, 0.25, 0.25], [0.5, 0.5, 1.0, 0.5, 0.25, 0.25]])
    expected_waves = np.array([100, 200, 300, 400, 500, 600])
    np.testing.assert_allclose(result_fluxes, expected_fluxes)
    np.testing.assert_allclose(result_wavelengths, expected_waves)

    # Test padding (this time WITH interpolation as well) when fluxes do not span the entire band
    b_band = create_test_passband(tmp_path, transmission_table, wave_grid=50, filter_name="b")
    fluxes = np.array([[0.25, 0.75, 1.0, 0.25]])
    flux_wavelengths = np.array([200.0, 300.0, 400.0, 500.0])
    (result_fluxes, result_wavelengths) = b_band._interpolate_flux_densities(
        fluxes, flux_wavelengths, "zeros"
    )
    # Check waves generated as expected
    expected_waves = np.arange(100, 601, 50)
    np.testing.assert_allclose(result_wavelengths, expected_waves)
    assert result_fluxes.shape == (1, 11)
    # Check flux zero-padding on both ends
    assert np.isclose(result_fluxes[0, 0], 0.0)
    assert np.isclose(result_fluxes[0, 1], 0.0)
    assert np.isclose(result_fluxes[0, 9], 0.0)
    assert np.isclose(result_fluxes[0, 10], 0.0)
    # Check the ends of the given fluxes are preserved
    assert np.isclose(result_fluxes[0, 2], 0.25)
    assert np.isclose(result_fluxes[0, 8], 0.25)
    # Check the increasing section (originally indices [0, 2]) is still increasing (now indices [2, 6])
    assert np.all(np.diff(result_fluxes[0, 2:6]) > 0)
    # Check the decreasing section (originally indices [2, 3]) is still decreasing (now indices [6, 8])
    assert np.all(np.diff(result_fluxes[0, 6:8]) < 0)


def test_passband_fluxes_to_bandflux(tmp_path):
    """Test the fluxes_to_bandflux method of the Passband class."""
    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    a_band = create_test_passband(tmp_path, transmission_table, wave_grid=100)

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
    expected_in_band_flux = np.trapz(
        flux * a_band.processed_transmission_table[:, 1], x=a_band.processed_transmission_table[:, 0]
    )
    in_band_flux = a_band.fluxes_to_bandflux(flux, a_band.processed_transmission_table[:, 0])
    np.testing.assert_allclose(in_band_flux, expected_in_band_flux)

    # Test with a different set of fluxes
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
    flux_wave_grid = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    a_band.set_transmission_table_grid(flux_wave_grid)
    in_band_flux = a_band.fluxes_to_bandflux(flux, flux_wave_grid)
    expected_in_band_flux = np.trapz(
        flux * a_band.processed_transmission_table[:, 1], x=a_band.processed_transmission_table[:, 0]
    )
    np.testing.assert_allclose(in_band_flux, expected_in_band_flux)

    # Test we raise an error if the fluxes are not the right shape
    with pytest.raises(ValueError):
        a_band.fluxes_to_bandflux(
            np.array([[1.0, 2.0], [3.0, 4.0]]), a_band.processed_transmission_table[:, 0]
        )

    # Test we raise an error if the fluxes are empty
    with pytest.raises(ValueError):
        a_band.fluxes_to_bandflux(np.array([]), a_band.processed_transmission_table[:, 0])
