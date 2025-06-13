import numpy as np
from tdastro.astro_utils.fake_sed_basis import FakeSEDBasis
from tdastro.astro_utils.passbands import Passband, PassbandGroup


def _create_toy_passbands() -> PassbandGroup:
    """Create a toy passband group with three passbands where the first passband
    has no overlap while the second two overlap each other for half the range.
    """
    a_band = Passband(np.array([[400, 0.5], [500, 0.5], [600, 0.5]]), "LSST", "u")
    b_band = Passband(np.array([[800, 0.8], [900, 0.8], [1000, 0.8]]), "LSST", "g")
    c_band = Passband(np.array([[900, 0.6], [1000, 0.6], [1100, 0.6]]), "LSST", "r")
    return PassbandGroup(given_passbands=[a_band, b_band, c_band])


def test_create_fake_sed_basis() -> None:
    """Test that we can create a simple FakeSEDBasis object."""
    pb_group = _create_toy_passbands()
    sed_basis = FakeSEDBasis.from_passbands(pb_group)

    # Check the internal structure of the FakeSEDBasis and that
    # when we pass each basis through the corresponding passband,
    # we get the expected SED values (1.0).
    assert len(sed_basis) == 3
    for filt in ["u", "g", "r"]:
        assert filt in sed_basis.sed_values
        assert len(sed_basis.sed_values[filt]) == len(pb_group.waves)

        # Interpolate the SED basis to the sample wavelengths.
        sampled_sed = sed_basis.get_basis(filt, pb_group.waves)
        assert len(sampled_sed) == len(pb_group.waves)

        # Check that the SED values are all 1.0 in the passband range.
        # Stack 3 copies s though we have three times.
        sampled_at_times = np.vstack([sampled_sed, sampled_sed, sampled_sed])
        sampled_bandflux = pb_group.fluxes_to_bandflux(sampled_at_times, filt)
        assert len(sampled_bandflux) == 3
        assert np.allclose(sampled_bandflux, 1.0)

    # Check that no two SED basis functions overlap.
    for f1 in ["u", "g", "r"]:
        sed1 = sed_basis.sed_values[f1]
        for f2 in ["u", "g", "r"]:
            if f1 != f2:
                sed2 = sed_basis.sed_values[f2]
                assert np.count_nonzero(sed1 * sed2) == 0


def test_create_single_fake_sed_basis() -> None:
    """Test that we can create a simple FakeSEDBasis object from a single passband."""
    a_band = Passband(np.array([[400, 0.5], [500, 0.5], [600, 0.5]]), "LSST", "u")
    sed_basis = FakeSEDBasis.from_passbands(a_band)
    assert len(sed_basis) == 1
    assert "u" in sed_basis.sed_values
    assert "g" not in sed_basis.sed_values
    assert "r" not in sed_basis.sed_values
