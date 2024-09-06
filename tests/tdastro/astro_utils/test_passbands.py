from unittest.mock import patch

import numpy as np
import pytest
from tdastro.astro_utils.passbands import Passband, PassbandGroup


def test_passband_init():
    """Test the initialization of the Passband class."""
    # Test that we can create an empty PassbandGroup object
    empty_passband = PassbandGroup()
    assert len(empty_passband.passbands) == 0

    # Test that the PassbandGroup class can be initialized with a preset
    lsst_passband_group = PassbandGroup(preset="LSST")
    assert len(lsst_passband_group.passbands) == 6
    assert "u" in lsst_passband_group.passbands
    assert "g" in lsst_passband_group.passbands
    assert "r" in lsst_passband_group.passbands
    assert "i" in lsst_passband_group.passbands
    assert "z" in lsst_passband_group.passbands
    assert "y" in lsst_passband_group.passbands

    # Test that the PassbandGroup class can be initialized with a list of Passband objects
    lsst_gri_passbands = [
        Passband("LSST", "g"),
        Passband("LSST", "r"),
        Passband("LSST", "i"),
    ]
    lsst_gri_passband_group = PassbandGroup(passbands=lsst_gri_passbands)
    assert len(lsst_gri_passband_group.passbands) == 3
    assert "g" in lsst_gri_passband_group.passbands
    assert "r" in lsst_gri_passband_group.passbands
    assert "i" in lsst_gri_passband_group.passbands

    # Test that the PassbandGroup class can be initialized with non-LSST passbands
    gaia_passbands = [
        Passband(
            "GAIA",
            "0.Gbp",
            table_url="http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id=GAIA/GAIA0.Gbp",
        ),
        Passband(
            "GAIA",
            "0.G",
            table_url="http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id=GAIA/GAIA0.G",
        ),
        Passband(
            "GAIA",
            "0.Grp",
            table_url="http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id=GAIA/GAIA0.Grp",
        ),
    ]
    gaia_passband_group = PassbandGroup(passbands=gaia_passbands)
    assert len(gaia_passband_group.passbands) == 3
    assert "0.Gbp" in gaia_passband_group.passbands
    assert "0.G" in gaia_passband_group.passbands
    assert "0.Grp" in gaia_passband_group.passbands

    # Test that the PassbandGroup class raises an error for an unknown preset
    try:
        _ = PassbandGroup(preset="Unknown")
    except ValueError as e:
        assert str(e) == "Unknown passband preset: Unknown"
    else:
        raise AssertionError("PassbandGroup should raise an error for an unknown preset")


def test_passband_str():
    """Test the __str__ method of the PassbandGroup class."""
    empty_passband_group = PassbandGroup()
    assert str(empty_passband_group) == "PassbandGroup containing 0 passbands: "

    lsst_passband_group = PassbandGroup(preset="LSST")
    assert (
        str(lsst_passband_group)
        == "PassbandGroup containing 6 passbands: LSST_u, LSST_g, LSST_r, LSST_i, LSST_z, LSST_y"
    )

    gaia_passbands = [
        Passband(
            "GAIA",
            "0.Gbp",
            table_url="http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id=GAIA/GAIA0.Gbp",
        ),
        Passband(
            "GAIA",
            "0.G",
            table_url="http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id=GAIA/GAIA0.G",
        ),
        Passband(
            "GAIA",
            "0.Grp",
            table_url="http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php?format=ascii&id=GAIA/GAIA0.Grp",
        ),
    ]
    gaia_passband_group = PassbandGroup(passbands=gaia_passbands)
    assert (
        str(gaia_passband_group) == "PassbandGroup containing 3 passbands: GAIA_0.Gbp, GAIA_0.G, GAIA_0.Grp"
    )


def test_passband_load_transmission_table(tmp_path):
    """Test the _load_transmission_table method of the Passband class."""
    survey = "TEST"
    band_label = "a"
    table_path = tmp_path / f"{survey}_{band_label}.dat"

    # Create a transmission table file
    transmission_table = "1000 0.5\n1005 0.6\n1010 0.7\n"
    with open(table_path, "w") as f:
        f.write(transmission_table)

    # Initialization of the Passband object automatically loads the transmission table
    a_band = Passband(survey, band_label, table_path=table_path)

    # Check that the transmission table was loaded correctly
    assert a_band.processed_transmission_table.shape == (3, 2)
    assert np.allclose(a_band.processed_transmission_table[:, 0], np.array([1000, 1005, 1010]))
    assert a_band.processed_transmission_table[0, 1] < a_band.processed_transmission_table[1, 1]
    assert a_band.processed_transmission_table[1, 1] < a_band.processed_transmission_table[2, 1]


def test_passband_download_transmission_table(tmp_path):
    """Test the _download_transmission_table method of the Passband class."""
    # Initialize a Passband object
    survey = "TEST"
    band_label = "a"
    table_path = tmp_path / f"{survey}_{band_label}.dat"
    table_url = f"http://example.com/{survey}/{band_label}.dat"
    transmission_table = "1000 0.5\n1005 0.6\n1010 0.7\n"

    def mock_urlretrieve(url, filename, *args, **kwargs):  # Urlretrieve saves contents directly to filename
        with open(filename, "w") as f:
            f.write(transmission_table)
        return filename, None

    # Mock the urlretrieve portion of the download method
    with patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve) as mocked_urlretrieve:
        a_band = Passband(survey, band_label, table_path=table_path, table_url=table_url)

        # Check that the transmission table was downloaded
        mocked_urlretrieve.assert_called_once_with(table_url, table_path)

        # Check that the transmission table was loaded correctly
        assert a_band.processed_transmission_table.shape == (3, 2)
        assert a_band.processed_transmission_table[0, 0] == 1000
        assert a_band.processed_transmission_table[1, 0] == 1005
        assert a_band.processed_transmission_table[2, 0] == 1010


def test_passband_interpolate_or_downsample_transmission_table(tmp_path):
    """TODO"""
    pass


def test_passband_normalize_interpolated_transmission_table(tmp_path):
    """Test the _normalize_interpolated_transmission_table method of the Passband class."""
    # Initialize a Passband object
    survey = "TEST"
    band_label = "a"
    table_path = tmp_path / f"{survey}_{band_label}.dat"

    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    with open(table_path, "w") as f:
        f.write(transmission_table)

    a_band = Passband(survey, band_label, table_path=table_path, wave_grid=100)

    # Compare results
    expected_result = np.array([[100.0, 0.0075], [200.0, 0.005625], [300.0, 0.00125]])
    assert np.allclose(a_band.processed_transmission_table, expected_result)


def test_set_transmission_table_to_new_grid(tmp_path):
    """TODO"""
    pass


def test_passband_interpolate_flux_densities_basic(tmp_path):
    """Test the _interpolate_flux_densities method of the Passband class. Make sure values are interpolated as
    expected."""
    # Initialize a Passband object
    survey = "TEST"
    band_label = "a"
    table_path = tmp_path / f"{survey}_{band_label}.dat"

    transmission_table = "100 0.5\n200 0.75\n300 0.25\n400 0.5\n500 0.25\n600 0.5"
    with open(table_path, "w") as f:
        f.write(transmission_table)

    # Test interpolation is skipped when not needed
    a_band = Passband(survey, band_label, table_path=table_path, wave_grid=100.0)
    fluxes = np.array([[0.25, 0.5, 1.0, 1.0, 0.5, 0.25]])
    flux_wavelengths = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    (result_fluxes, result_wavelengths) = a_band._interpolate_flux_densities(fluxes, flux_wavelengths)
    np.testing.assert_allclose(result_fluxes, fluxes)
    np.testing.assert_allclose(result_wavelengths, flux_wavelengths)

    # Test interpolation when fluxes span entire band
    b_band = Passband(survey, band_label, table_path=table_path, wave_grid=50.0)
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
    # First half of array should be increasing
    assert np.all(np.diff(result_fluxes[0, :6]) > 0)
    # Second half of array should be decreasing
    assert np.all(np.diff(result_fluxes[0, 5:]) < 0)


def test_passband_interpolate_flux_densities_mismatched_bounds(
    tmp_path,
):
    """Test the _interpolate_flux_densities method of the Passband class. Make sure values are interpolated as
    expected, out-of-bounds values are handled correctly, and padding is creating when needed."""
    # Initialize a Passband object
    survey = "TEST"
    band_label = "a"
    table_path = tmp_path / f"{survey}_{band_label}.dat"

    transmission_table = "100 0.5\n200 0.75\n300 0.25\n400 0.5\n500 0.25\n600 0.5"
    with open(table_path, "w") as f:
        f.write(transmission_table)

    # Test interpolation when fluxes are out of bounds
    a_band = Passband(survey, band_label, table_path=table_path, wave_grid=100.0)
    fluxes = np.array([[0.25, 0.5, 1.0, 1.0, 0.5, 0.25]])
    flux_wavelengths = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 650.0])

    # Test padding (with no interpolation) when fluxes do not span the entire band
    a_band = Passband(survey, band_label, table_path=table_path, wave_grid=100.0)
    fluxes = np.array([[0.5, 1.0, 0.5, 0.25], [0.5, 1.0, 0.5, 0.25]])
    flux_wavelengths = np.array([200.0, 300.0, 400.0, 500.0])

    # Case 1: with extrapolation="raise"
    with pytest.raises(ValueError):
        a_band._interpolate_flux_densities(fluxes, flux_wavelengths, "raise")

    # Case 2: with extrapolation="zeros"
    (result_fluxes, result_wavelengths) = a_band._interpolate_flux_densities(
        fluxes, flux_wavelengths, "zeros"
    )
    expected_fluxes = np.array([[0.0, 0.5, 1.0, 0.5, 0.25, 0.0], [0.0, 0.5, 1.0, 0.5, 0.25, 0.0]])
    expected_waves = np.array([100, 200, 300, 400, 500, 600])
    np.testing.assert_allclose(result_fluxes, expected_fluxes)
    np.testing.assert_allclose(result_wavelengths, expected_waves)

    # Case 3: with extrapolation="constant"
    (result_fluxes, result_wavelengths) = a_band._interpolate_flux_densities(
        fluxes, flux_wavelengths, "const"
    )
    expected_fluxes = np.array([[0.5, 0.5, 1.0, 0.5, 0.25, 0.25], [0.5, 0.5, 1.0, 0.5, 0.25, 0.25]])
    expected_waves = np.array([100, 200, 300, 400, 500, 600])
    np.testing.assert_allclose(result_fluxes, expected_fluxes)
    np.testing.assert_allclose(result_wavelengths, expected_waves)

    # Test interpolation and padding when fluxes do not span the entire band
    d_band = Passband(survey, band_label, table_path=table_path, wave_grid=50.0)
    fluxes = np.array([[0.25, 0.75, 1.0, 0.25]])
    flux_wavelengths = np.array([200.0, 300.0, 400.0, 500.0])
    (result_fluxes, result_wavelengths) = d_band._interpolate_flux_densities(
        fluxes, flux_wavelengths, "zeros"
    )
    expected_waves = np.arange(100, 601, 50)
    np.testing.assert_allclose(result_wavelengths, expected_waves)
    assert result_fluxes.shape == (1, 11)
    # Check zero-padding
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

    # Test handling when fluxes are out of bounds
    e_band = Passband(survey, band_label, table_path=table_path, wave_grid=100.0)
    fluxes = np.array([[0.25, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, 0.25]])
    flux_wavelengths = np.array([50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 650.0])
    (result_fluxes, result_wavelengths) = e_band._interpolate_flux_densities(fluxes, flux_wavelengths)
    expected_waves = np.arange(100, 601, 100)
    np.testing.assert_allclose(result_wavelengths, expected_waves)
    np.testing.assert_allclose(result_fluxes, fluxes[:, 1:7])


def test_passband_fluxes_to_bandflux(tmp_path):
    """Test the fluxes_to_bandflux method of the Passband class."""
    # Initialize a Passband object
    survey = "TEST"
    band_label = "a"
    table_path = tmp_path / f"{survey}_{band_label}.dat"

    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    with open(table_path, "w") as f:
        f.write(transmission_table)

    a_band = Passband(survey, band_label, table_path=table_path, wave_grid=100.0)

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

    # Compare results
    in_band_flux = a_band.fluxes_to_bandflux(flux, a_band.processed_transmission_table[:, 0])
    np.allclose(in_band_flux, expected_in_band_flux)


def test_passband_group_fluxes_to_bandfluxes(tmp_path):
    """Test the fluxes_to_bandfluxes method of the PassbandGroup class."""
    # Initialize requirements for PassbandGroup object
    survey = "TEST"
    band_labels = ["a", "b", "c"]
    table_dir = tmp_path / survey
    table_dir.mkdir()

    transmission_tables = {
        "a": "100 0.5\n200 0.75\n300 0.25\n",
        "b": "100 0.25\n200 0.5\n300 0.75\n",
        "c": "100 0.75\n200 0.25\n300 0.5\n",
    }
    for band_label in band_labels:
        with open(table_dir / f"{band_label}.dat", "w") as f:
            print("Writing", band_label, "to", table_dir / f"{band_label}.dat")
            f.write(transmission_tables[band_label])

    # Load the PassbandGroup object
    passbands = []
    for band_label in band_labels:
        passbands.append(
            Passband(survey, band_label, table_path=table_dir / f"{band_label}.dat", wave_grid=100.0)
        )
    test_passband_group = PassbandGroup(passbands=passbands)

    # Define some mock flux values and calculate our expected bandfluxes
    flux = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    flux_wavelengths = np.array([100, 200, 300])

    # Compare results
    bandfluxes = test_passband_group.fluxes_to_bandfluxes(flux, flux_wavelengths)
    for band_name in test_passband_group.passbands:
        assert band_name in bandfluxes
        assert bandfluxes[band_name].shape == (5,)
        assert np.allclose(
            bandfluxes[band_name],
            np.trapz(
                flux * test_passband_group.passbands[band_name].processed_transmission_table[:, 1],
                x=flux_wavelengths,
                axis=1,
            ),
        )


def test_passbandgroup_set_transmission_table_grids(tmp_path):
    """Test the set_wave_grid_att method of the PassbandGroup class. Note this"""
    # Initialize a Passband object
    survey = "TEST"
    band_label = "a"
    table_path = tmp_path / f"{survey}_{band_label}.dat"

    transmission_table = "100 0.5\n200 0.75\n300 0.25\n400 0.5\n500 0.25\n600 0.5"
    with open(table_path, "w") as f:
        f.write(transmission_table)

    # TODO update to match set_transmission_table_grids

    # # Test interpolation is skipped when not needed
    # a_band = Passband(survey, band_label, table_path=table_path, wave_grid=100.0)
    # np.testing.assert_allclose(a_band.wave_grid, np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]))
    # np.testing.assert_allclose(
    #     a_band.processed_transmission_table[:, 0],
    #     np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
    # )

    # # Test that grid is reset successfully (but the transmission table is not yet interpolated)
    # a_band._set_wave_grid_attr(50.0)
    # np.testing.assert_allclose(a_band.wave_grid, np.arange(100.0, 601.0, 50.0))
    # np.testing.assert_allclose(
    #     a_band.processed_transmission_table[:, 0],
    #     np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
    # )

    # # Test we can handle a list as input
    # a_band._set_wave_grid_attr([100, 200, 300, 400, 500, 600])
    # np.testing.assert_allclose(a_band.wave_grid, np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]))

    # # Test we can handle an int as input
    # a_band._set_wave_grid_attr(50)
    # np.testing.assert_allclose(a_band.wave_grid, np.arange(100.0, 601.0, 50.0))


def test_passbandgroup_sreset_transmission_table_grid(tmp_path):
    """Test the set_transmission_table_to_new_grid method of the PassbandGroup class."""
    # Initialize a Passband object
    survey = "TEST"
    band_label = "a"
    table_path = tmp_path / f"{survey}_{band_label}.dat"

    transmission_table = "100 0.5\n200 0.75\n300 0.25\n400 0.5\n500 0.25\n600 0.5"
    with open(table_path, "w") as f:
        f.write(transmission_table)

    # Test interpolation is skipped when not needed
    a_band = Passband(survey, band_label, table_path=table_path, wave_grid=100.0)
    np.testing.assert_allclose(a_band.wave_grid, np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]))
    np.testing.assert_allclose(
        a_band.processed_transmission_table[:, 0],
        np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0]),
    )

    # Test that grid is reset successfully AND we have interpolated the transmission table as well
    a_band.set_transmission_table_grid(50.0)
    np.testing.assert_allclose(a_band.wave_grid, np.arange(100.0, 601.0, 50.0))
    np.testing.assert_allclose(
        a_band.processed_transmission_table[:, 0],
        np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0]),
    )
