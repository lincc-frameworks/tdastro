from tdastro.astro_utils.passbands import Passband, PassbandGroup


def test_passband_init():
    """Test the initialization of the Passband class."""
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

    empty_passband_group = PassbandGroup()
    assert str(empty_passband_group) == "PassbandGroup containing 0 passbands: "
