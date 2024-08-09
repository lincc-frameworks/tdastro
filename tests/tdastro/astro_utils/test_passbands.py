import os
import urllib
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tdastro.astro_utils.passbands import Passbands
from tdastro.sources.spline_model import SplineModel


def test_passbands_init():
    """Test that we can initialize a Passbands object with different bands."""
    passbands_default = Passbands()
    assert passbands_default.bands == []

    passbands_ubv = Passbands(bands=["u", "b", "v"])
    assert passbands_ubv.bands == ["u", "b", "v"]


def test_passbands_get_data_path(tmp_path):
    """Test that we can find/create a data directory for the bandpass data."""
    passbands = Passbands()

    # Check that the data path is empty before we call the method
    assert passbands.data_path is None

    # Mock the os.path.join function to point to our tmp path
    tmp_data_path = os.path.join(tmp_path, "band_data/")
    with patch("os.path.join") as mocked_join:
        mocked_join.return_value = tmp_data_path
        passbands._get_data_path()

        assert os.path.isdir(passbands.data_path)
        assert os.path.isdir(Path(f"{tmp_path}/band_data/"))
        assert Path(passbands.data_path) == Path(f"{tmp_path}/band_data/")


def test_passbands_load_local_transmission_table():
    """Test that we can load a transmission table from a file that already exists."""
    expected_data = np.array([[100.0, 0.5], [200.0, 0.75], [300.0, 0.25]])
    with patch("numpy.loadtxt", return_value=expected_data):
        passbands = Passbands(bands="test-band")
        passbands._load_transmission_table("test-band", "mock_file.dat")
        np.testing.assert_array_equal(passbands.transmission_tables["test-band"], expected_data)


def test_passbands_download_transmission_table():
    """Test that we can download a transmission table and load the file contents."""
    with patch("urllib.request.urlretrieve", return_value=True):
        with patch("os.path.getsize", return_value=42):  # Just needs to be non-empty
            passbands = Passbands(bands="test-band")
            assert passbands._download_transmission_table("test-band", "mock_file.dat")


def test_passbands_download_transmission_table_http_error():
    """Test that we handle HTTPError when downloading a transmission table."""
    passbands = Passbands(bands=["test-band"])
    with patch(
        "urllib.request.urlretrieve",
        side_effect=urllib.error.HTTPError(url="", code=404, msg="Not Found", hdrs=None, fp=None),
    ):
        assert not passbands._download_transmission_table("test-band", "mock_file.dat")


def test_passbands_download_transmission_table_url_error():
    """Test that we handle URLError when downloading a transmission table."""
    passbands = Passbands(bands=["test-band"])
    with patch("urllib.request.urlretrieve", side_effect=urllib.error.URLError("No host")):
        assert not passbands._download_transmission_table("test-band", "mock_file.dat")


def test_passbands_load_transmission_table_value_error():
    """Test that we handle ValueError when loading a transmission table."""
    passbands = Passbands(bands=["test-band"])
    with patch("numpy.loadtxt", side_effect=ValueError("Mocked ValueError")):
        assert not passbands._load_transmission_table("test-band", "mock_file.dat")


def test_passbands_load_transmission_table_os_error():
    """Test that we handle OSError when loading a transmission table."""
    passbands = Passbands(bands=["test-band"])
    with patch("numpy.loadtxt", side_effect=OSError("Mocked OSError")):
        assert not passbands._load_transmission_table("test-band", "mock_file.dat")


def test_passbands_load_all_transmission_tables(tmp_path):
    """Test that we can download and load all transmission tables."""
    band_ids = ["u", "g"]
    passbands = Passbands(bands=band_ids)

    # Mock the _get_data_path method
    with patch.object(passbands, "_get_data_path"):
        passbands.data_path = os.path.join(tmp_path, "band_data/")

        # Mock the _download_transmission_table and _load_transmission_table methods
        with patch.object(passbands, "_download_transmission_table", return_value=True), patch.object(
            passbands, "_load_transmission_table", return_value=True
        ):
            passbands.load_all_transmission_tables()

            # Assert _download_transmission_table was called for each band
            assert passbands._download_transmission_table.call_count == len(band_ids)

            # Assert _load_transmission_table was called for each band
            assert passbands._load_transmission_table.call_count == len(band_ids)


def test_passbands_phi_b():
    """Test that we can calculate the value of phi_b for all wavelengths in a transmission table."""
    passbands = Passbands()

    u_band_transmission_table = np.array([[100.0, 0.5], [200.0, 0.75], [300.0, 0.25]])
    g_band_transmission_table = np.array([[100.0, 0.75], [200.0, 0.5], [300.0, 0.25]])

    u_band_phi_b = passbands._phi_b(u_band_transmission_table)
    g_band_phi_b = passbands._phi_b(g_band_transmission_table)

    assert len(u_band_phi_b) == len(u_band_transmission_table)
    assert len(g_band_phi_b) == len(g_band_transmission_table)

    # Check that the integrated value of u_band_phi_b about-equals that of g_band_phi_b
    assert np.isclose(
        np.trapz(u_band_phi_b, x=u_band_transmission_table[:, 0]),
        np.trapz(g_band_phi_b, x=g_band_transmission_table[:, 0]),
        rtol=1e-9,  # relative tolerance
        atol=1e-9,  # absolute tolerance
    )

    # And that they're both about equal to 1.0
    assert np.isclose(np.trapz(u_band_phi_b, x=u_band_transmission_table[:, 0]), 1.0, rtol=1e-9, atol=1e-9)
    assert np.isclose(np.trapz(g_band_phi_b, x=g_band_transmission_table[:, 0]), 1.0, rtol=1e-9, atol=1e-9)

    # Check that they match our hand-computed values
    np.testing.assert_allclose(u_band_phi_b, [0.0075, 0.005625, 0.00125], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(g_band_phi_b, [0.01125, 0.00375, 0.00125], rtol=1e-9, atol=1e-9)


def test_passbands_calculate_normalized_system_response_tables():
    """Test that we can calculate the normalized system response tables for all bands."""
    band_ids = ["u", "g"]
    passbands = Passbands(bands=band_ids)

    # Add mock transmission tables
    passbands.transmission_tables["u"] = np.array([[100.0, 0.5], [200.0, 0.75], [300.0, 0.25]])
    passbands.transmission_tables["g"] = np.array([[100.0, 0.75], [200.0, 0.5], [300.0, 0.25]])

    # Expected results
    expected_results = {
        "u": np.array([[100.0, 0.0075], [200.0, 0.005625], [300.0, 0.00125]]),
        "g": np.array([[100.0, 0.01125], [200.0, 0.00375], [300.0, 0.00125]]),
    }

    # Calculate the normalized system response tables
    passbands.calculate_normalized_system_response_tables()

    # Check we have not computed too many/too few bands
    assert len(passbands.normalized_system_response_tables) == len(band_ids)

    # Check that the normalized system response tables have been calculated for each band
    for band_id in band_ids:
        assert band_id in passbands.normalized_system_response_tables  # band_id is in the dictionary
        assert passbands.normalized_system_response_tables[band_id].shape == (3, 2)  # 3 rows, 2 columns
        np.testing.assert_allclose(
            passbands.normalized_system_response_tables[band_id][:, 1],
            expected_results[band_id][:, 1],
            rtol=1e-9,
            atol=1e-9,
        )


def test_passbands_get_in_band_flux():
    """Test the calculation of in-band flux for given flux and normalized system response table."""
    # Set up the passbands object
    passbands = Passbands(bands=["test-band"])
    passbands.transmission_tables["test-band"] = np.array([[100.0, 0.5], [200.0, 0.75], [300.0, 0.25]])
    passbands.calculate_normalized_system_response_tables()

    # Define some mock flux values
    flux = np.array([1.0, 2.0, 3.0])

    # Calculate the in-band flux we expect
    normalized_system_response_table = passbands.normalized_system_response_tables["test-band"]
    expected_in_band_flux = np.trapz(
        flux * normalized_system_response_table[:, 1], x=normalized_system_response_table[:, 0]
    )

    # Calculate in-band flux with target method
    calculated_in_band_flux = passbands._get_in_band_flux(flux, "test-band")

    # Check that the calculated in-band flux is correct
    assert np.isclose(calculated_in_band_flux, expected_in_band_flux, rtol=1e-9, atol=1e-9)

    # Check against a hand-computed value; note that this will change if we alter parameters above
    assert np.isclose(calculated_in_band_flux, 1.6875, rtol=1e-5, atol=1e-5)


def test_passbands_generate_all_in_band_fluxes():
    """Test that we can generate the in-band fluxes for all bands given a SplineModel and times.

    Check initially for a flat spectrum model, where we can expect our colors to be equivalent; then
    check for a non-flat spectrum model (where colors are not equivalent)."""

    passbands = Passbands(bands=["a", "b", "c"])
    passbands.transmission_tables["a"] = np.array([[100.0, 0.5], [200.0, 0.75]])
    passbands.transmission_tables["b"] = np.array([[200.0, 0.25], [300.0, 0.5]])
    passbands.transmission_tables["c"] = np.array([[300.0, 0.5], [400.0, 0.5]])
    passbands.calculate_normalized_system_response_tables()

    # Define some mock times
    times = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    wavelengths = np.array([100.0, 200.0, 300.0, 400.0])

    # Create our model
    model = SplineModel(
        times,
        wavelengths,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0, 5.0],
                [9.0, 9.0, 9.0, 9.0],
                [8.0, 8.0, 8.0, 8.0],
                [7.0, 7.0, 7.0, 7.0],
            ]
        ),
        time_degree=1,
        wave_degree=1,
    )

    # Calculate in-band fluxes
    calculated_flux_matrix = passbands.generate_in_band_fluxes(model, times)

    # Check that the shape of the calculated flux matrix is correct
    assert calculated_flux_matrix.shape == (5, 3)

    # Check that the colors are equivalent for a flat spectrum model
    # That is, that column B - column A == column C - column B
    np.testing.assert_allclose(
        calculated_flux_matrix[:, 2] - calculated_flux_matrix[:, 1],
        calculated_flux_matrix[:, 1] - calculated_flux_matrix[:, 0],
        rtol=1e-9,
        atol=1e-9,
    )

    # Check with a non-flat spectrum model
    model_b = SplineModel(
        times,
        wavelengths,
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 9.0, 18.0, 9.0],
                [8.0, 6.0, 4.0, 6.0],
                [7.0, 0.0, 1.0, 1.0],
            ]
        ),
        time_degree=3,
        wave_degree=3,
    )

    # Check that we can successfully run the method with a non-flat spectrum model
    calculated_flux_matrix_b = passbands.generate_in_band_fluxes(model_b, times)

    # Check that the shape of the calculated flux matrix is correct
    assert calculated_flux_matrix_b.shape == (5, 3)

    # Check that the colors are not equivalent for a non-flat spectrum model such as this
    assert not np.allclose(
        calculated_flux_matrix_b[:, 2] - calculated_flux_matrix_b[:, 1],
        calculated_flux_matrix_b[:, 1] - calculated_flux_matrix_b[:, 0],
        rtol=1e-9,
        atol=1e-9,
    )


def test_convert_fluxes_to_in_band_fluxes():
    """Test that we can convert fluxes to in-band fluxes."""
    passbands = Passbands(bands=["a", "b", "c"])
    passbands.transmission_tables["a"] = np.array([[100.0, 0.5], [200.0, 0.75]])
    passbands.transmission_tables["b"] = np.array([[200.0, 0.25], [300.0, 0.5]])
    passbands.transmission_tables["c"] = np.array([[300.0, 0.5], [400.0, 0.5]])
    passbands.calculate_normalized_system_response_tables()

    # Define some mock flux values
    fluxes = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 9.0, 8.0],
            [8.0, 8.0, 8.0, 8.0],
            [7.0, 8.0, 9.0, 10.0],
        ]
    )
    wavelengths = np.array([100.0, 200.0, 300.0, 400.0])

    # Calculate the in-band fluxes
    in_band_fluxes = passbands.convert_fluxes_to_in_band_fluxes(wavelengths, fluxes)

    # Check that the shape of the in-band fluxes matrix is correct
    assert in_band_fluxes.shape == (5, 3)
