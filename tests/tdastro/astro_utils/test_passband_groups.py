from unittest.mock import patch

import numpy as np
from tdastro.astro_utils.passbands import Passband, PassbandGroup


def create_passband_group(path):
    """Helper function to create a PassbandGroup object for testing."""
    # Initialize requirements for PassbandGroup object
    survey = "TEST"
    band_labels = ["a", "b", "c"]
    table_dir = path / survey
    table_dir.mkdir()

    transmission_tables = {
        "a": "100 0.5\n200 0.75\n300 0.25\n",
        "b": "100 0.25\n200 0.5\n300 0.75\n",
        "c": "100 0.75\n200 0.25\n300 0.5\n",
    }
    for band_label in band_labels:
        with open(table_dir / f"{band_label}.dat", "w") as f:
            f.write(transmission_tables[band_label])

    # Load the PassbandGroup object
    passbands = []
    for band_label in band_labels:
        passbands.append(
            Passband(survey, band_label, table_path=table_dir / f"{band_label}.dat", wave_grid=100.0)
        )
    return PassbandGroup(passbands=passbands)


def test_passband_group_init():
    """Test the initialization of the Passband class, and implicitly, _load_preset."""
    # Test that we can create an empty PassbandGroup object
    empty_passband = PassbandGroup()
    assert len(empty_passband.passbands) == 0

    # Test that the PassbandGroup class can be initialized with a preset
    # Mock the transmission table files at passbands/LSST/<filter>.dat using patch
    transmission_table_array = np.array([[100, 0.5], [200, 0.75], [300, 0.25]])

    def mock_load_transmission_table(self):
        self._loaded_table = transmission_table_array

    with patch.object(Passband, "_load_transmission_table", new=mock_load_transmission_table):
        lsst_passband_group = PassbandGroup(preset="LSST")
        assert len(lsst_passband_group.passbands) == 6
        assert "LSST_u" in lsst_passband_group.passbands
        assert "LSST_g" in lsst_passband_group.passbands
        assert "LSST_r" in lsst_passband_group.passbands
        assert "LSST_i" in lsst_passband_group.passbands
        assert "LSST_z" in lsst_passband_group.passbands
        assert "LSST_y" in lsst_passband_group.passbands

    # Test that the PassbandGroup class can be initialized with a list of Passband objects
    with patch.object(Passband, "_load_transmission_table", new=mock_load_transmission_table):
        lsst_gri_passbands = [
            Passband("LSST", "g"),
            Passband("LSST", "r"),
            Passband("LSST", "i"),
        ]
        lsst_gri_passband_group = PassbandGroup(passbands=lsst_gri_passbands)
        assert len(lsst_gri_passband_group.passbands) == 3
        assert "LSST_g" in lsst_gri_passband_group.passbands
        assert "LSST_r" in lsst_gri_passband_group.passbands
        assert "LSST_i" in lsst_gri_passband_group.passbands

    # Test that the PassbandGroup class raises an error for an unknown preset
    try:
        _ = PassbandGroup(preset="Unknown")
    except ValueError as e:
        assert str(e) == "Unknown passband preset: Unknown"
    else:
        raise AssertionError("PassbandGroup should raise an error for an unknown preset")


def test_passband_group_str(tmp_path):
    """Test the __str__ method of the PassbandGroup class."""
    empty_passband_group = PassbandGroup()
    assert str(empty_passband_group) == "PassbandGroup containing 0 passbands: "

    # Test that the __str__ method returns the expected string with preset passbands
    lsst_passband_group = PassbandGroup(preset="LSST")
    assert (
        str(lsst_passband_group)
        == "PassbandGroup containing 6 passbands: LSST_u, LSST_g, LSST_r, LSST_i, LSST_z, LSST_y"
    )

    # Test that the __str__ method returns the expected string with custom passbands
    test_passband_group = create_passband_group(tmp_path)
    assert str(test_passband_group) == "PassbandGroup containing 3 passbands: TEST_a, TEST_b, TEST_c"


def test_passband_group_set_transmission_table_grid(tmp_path):
    """Test the set_transmission_table_to_new_grid method of the PassbandGroup class."""
    test_passband_group = create_passband_group(tmp_path)

    # Test that grid is reset successfully AND we have called interpolation methods the transmission table
    with patch.object(Passband, "set_transmission_table_grid") as mock_set_grid:
        # Call the method with a float
        test_passband_group.set_transmission_table_grids(50.0)

        # Check that the method was called for each Passband object
        assert mock_set_grid.call_count == len(test_passband_group.passbands)
        for _ in test_passband_group.passbands:
            mock_set_grid.assert_any_call(50.0)

        # Call the method with None
        test_passband_group.set_transmission_table_grids(None)

        # Check that the method was called for each Passband object
        assert mock_set_grid.call_count == 2 * len(test_passband_group.passbands)
        for _ in test_passband_group.passbands:
            mock_set_grid.assert_any_call(None)


def test_passband_group_fluxes_to_bandfluxes(tmp_path):
    """Test the fluxes_to_bandfluxes method of the PassbandGroup class."""
    # Test with empty passband group
    empty_passband_group = PassbandGroup()
    flux = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
    flux_wavelengths = np.array([100, 200, 300])
    bandfluxes = empty_passband_group.fluxes_to_bandfluxes(flux, flux_wavelengths)
    assert len(bandfluxes) == 0

    # Test with simple passband group
    test_passband_group = create_passband_group(tmp_path)
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
    bandfluxes = test_passband_group.fluxes_to_bandfluxes(flux, flux_wavelengths)

    # Check values are as expected
    assert len(bandfluxes) == 3
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
