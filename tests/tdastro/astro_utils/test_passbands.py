from unittest.mock import patch

import numpy as np
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
    transmission_table = "1000 0.5\n1001 0.6\n1002 0.7\n"
    with open(table_path, "w") as f:
        f.write(transmission_table)

    # Initialization of the Passband object automatically loads the transmission table
    a_band = Passband(survey, band_label, table_path=table_path)

    # Check that the transmission table was loaded correctly
    assert a_band.transmission_table.shape == (3, 2)
    assert a_band.transmission_table[0, 0] == 1000
    assert a_band.transmission_table[0, 1] == 0.5
    assert a_band.transmission_table[1, 0] == 1001
    assert a_band.transmission_table[1, 1] == 0.6
    assert a_band.transmission_table[2, 0] == 1002
    assert a_band.transmission_table[2, 1] == 0.7


def test_passband_download_transmission_table(tmp_path):
    """Test the _download_transmission_table method of the Passband class."""
    # Initialize a Passband object
    survey = "TEST"
    band_label = "a"
    table_path = tmp_path / f"{survey}_{band_label}.dat"
    table_url = f"http://example.com/{survey}/{band_label}.dat"
    transmission_table = "1000 0.5\n1001 0.6\n1002 0.7\n"

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
        assert a_band.transmission_table.shape == (3, 2)
        assert a_band.transmission_table[0, 0] == 1000
        assert a_band.transmission_table[0, 1] == 0.5
        assert a_band.transmission_table[1, 0] == 1001
        assert a_band.transmission_table[1, 1] == 0.6
        assert a_band.transmission_table[2, 0] == 1002
        assert a_band.transmission_table[2, 1] == 0.7


def test_passband_normalize_transmission_table(tmp_path):
    """Test the _normalize_transmission_table method of the Passband class."""
    # Initialize a Passband object
    survey = "TEST"
    band_label = "a"
    table_path = tmp_path / f"{survey}_{band_label}.dat"

    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    with open(table_path, "w") as f:
        f.write(transmission_table)

    a_band = Passband(survey, band_label, table_path=table_path)

    # Compare results
    expected_result = np.array([[100.0, 0.0075], [200.0, 0.005625], [300.0, 0.00125]])
    assert np.allclose(a_band.normalized_transmission_table, expected_result)


def test_passband_fluxes_to_bandflux(tmp_path):
    """Test the fluxes_to_bandflux method of the Passband class."""
    # Initialize a Passband object
    survey = "TEST"
    band_label = "a"
    table_path = tmp_path / f"{survey}_{band_label}.dat"

    transmission_table = "100 0.5\n200 0.75\n300 0.25\n"
    with open(table_path, "w") as f:
        f.write(transmission_table)

    a_band = Passband(survey, band_label, table_path=table_path)

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
        flux * a_band.normalized_transmission_table[:, 1], x=a_band.normalized_transmission_table[:, 0]
    )

    # Compare results
    in_band_flux = a_band.fluxes_to_bandflux(flux, a_band.normalized_transmission_table[:, 0])
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
        passbands.append(Passband(survey, band_label, table_path=table_dir / f"{band_label}.dat"))
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
    wavelengths = np.array([100, 200, 300])

    # Compare results
    bandfluxes = test_passband_group.fluxes_to_bandfluxes(flux, wavelengths)
    for label in test_passband_group.passbands:
        assert label in bandfluxes
        assert bandfluxes[label].shape == (5,)
        assert np.allclose(
            bandfluxes[label],
            np.trapz(
                flux * test_passband_group.passbands[label].normalized_transmission_table[:, 1],
                x=wavelengths,
                axis=1,
            ),
        )
