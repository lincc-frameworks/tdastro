from unittest.mock import patch

import numpy as np
from tdastro.astro_utils.passbands import Passband, PassbandGroup


def create_passband_group(path, delta_wave=5.0, trim_percentile=None):
    """Helper function to create a PassbandGroup object for testing."""
    # Initialize requirements for PassbandGroup object
    survey = "TEST"
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
                "trim_percentile": trim_percentile,
            }
        )
    return PassbandGroup(passband_parameters=passbands)


def test_passband_group_init(tmp_path):
    """Test the initialization of the Passband class, and implicitly, _load_preset."""
    # Test that we can create an empty PassbandGroup object
    empty_passband = PassbandGroup()
    assert len(empty_passband.passbands) == 0

    # Test that the PassbandGroup class can be initialized with a preset
    # Mock the transmission table files at passbands/LSST/<filter>.dat using patch
    transmission_table_array = np.array([[100, 0.5], [200, 0.75], [300, 0.25]])

    def mock_load_transmission_table(self, **kwargs):
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
            {"survey": "LSST", "filter_name": "g"},
            {"survey": "LSST", "filter_name": "r"},
            {"survey": "LSST", "filter_name": "i"},
        ]
        lsst_gri_passband_group = PassbandGroup(passband_parameters=lsst_gri_passbands)
        assert len(lsst_gri_passband_group.passbands) == 3
        assert "LSST_g" in lsst_gri_passband_group.passbands
        assert "LSST_r" in lsst_gri_passband_group.passbands
        assert "LSST_i" in lsst_gri_passband_group.passbands

    # Test that our helper function creates a PassbandGroup object with the expected passbands
    test_passband_group = create_passband_group(tmp_path)
    assert len(test_passband_group.passbands) == 3
    assert "TEST_a" in test_passband_group.passbands
    assert "TEST_b" in test_passband_group.passbands
    assert "TEST_c" in test_passband_group.passbands
    assert np.allclose(
        test_passband_group.waves,
        np.unique(np.concatenate([np.arange(100, 301, 5), np.arange(250, 351, 5), np.arange(400, 601, 5)])),
    )

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


def test_passband_group_fluxes_to_bandfluxes(tmp_path):
    """Test the fluxes_to_bandfluxes method of the PassbandGroup class."""
    # Test with simple passband group
    test_passband_group = create_passband_group(tmp_path, delta_wave=20, trim_percentile=None)

    # Create some mock flux data with the same number of columns as wavelengths in our PassbandGroup
    flux = np.linspace(test_passband_group.waves * 10, 100, 5)

    bandfluxes = test_passband_group.fluxes_to_bandfluxes(flux)

    # Check structure of result is as expected
    assert len(bandfluxes) == 3
    for band_name in test_passband_group.passbands:
        assert band_name in bandfluxes
        assert bandfluxes[band_name].shape == (5,)
