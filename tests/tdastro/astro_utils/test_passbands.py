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
    assert passbands_default.bands == ["u", "g", "r", "i", "z", "y"]

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


def test_passbands_calculate_normalized_system_response_tables():
    """Test that we can calculate the normalized system response tables for all bands."""
    band_ids = ["u", "g"]
    passbands = Passbands(bands=band_ids)

    # Add mock transmission tables
    passbands.transmission_tables["u"] = np.array([[100.0, 0.5], [200.0, 0.75], [300.0, 0.25]])
    passbands.transmission_tables["g"] = np.array([[100.0, 0.75], [200.0, 0.5], [300.0, 0.25]])

    # Mock the _phi_b method
    with patch.object(passbands, "_phi_b", return_value=np.array([0.5, 0.5, 0.5])):
        passbands.calculate_normalized_system_response_tables()

        # Check that the normalized system response tables have been calculated for each band
        for band_id in band_ids:
            assert band_id in passbands.normalized_system_response_tables  # band_id is in the dictionary
            assert passbands.normalized_system_response_tables[band_id].shape == (3, 2)  # 3 rows, 2 columns
            np.testing.assert_allclose(
                passbands.normalized_system_response_tables[band_id][:, 1], 0.5
            )  # vals = 0.5


def test_passbands_get_in_band_flux():
    """Test the calculation of in-band flux for given flux and normalized system response table."""
    passbands = Passbands(bands=["test-band"])

    # Mock transmission table data for test-band
    normalized_system_response_table = np.array([[100.0, 0.5], [200.0, 0.75], [300.0, 0.25]])
    flux = np.array([1.0, 2.0, 3.0])

    expected_in_band_flux = np.trapz(
        flux * normalized_system_response_table[:, 1], x=normalized_system_response_table[:, 0]
    )

    # Calculate in-band flux using the method
    calculated_in_band_flux = passbands._get_in_band_flux(flux, normalized_system_response_table)

    np.testing.assert_allclose(calculated_in_band_flux, expected_in_band_flux, rtol=1e-9, atol=1e-9)


def test_passbands_get_all_in_band_fluxes():
    """Test that we can calculate the in-band fluxes for all bands given a SplineModel and times."""

    passbands = Passbands(bands=["a", "b"])

    # Mock the normalized system response table for test-band
    passbands.normalized_system_response_tables["a"] = np.array([[100.0, 0.5], [200.0, 0.75], [300.0, 0.25]])
    passbands.normalized_system_response_tables["b"] = np.array([[100.0, 0.75], [200.0, 0.5], [300.0, 0.25]])

    # Define some mock times
    times = np.array([0.1, 0.2, 0.3])

    # Create our model
    model = SplineModel(
        times,
        [100.0, 200.0, 300.0],
        np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        time_degree=1,
        wave_degree=1,
    )

    # Calculate in-band fluxes using the method
    calculated_flux_matrix = passbands.get_all_in_band_fluxes(model, times)

    print(calculated_flux_matrix)  # Hmm...to discuss tomorrow (TODO)
