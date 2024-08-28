import numpy as np
from astropy import units as u
from tdastro.astro_utils.unit_utils import flam_to_fnu


def test_flam_to_fnu():
    """Test flam to fnu conversion
    fnu = flam * lam**2 / c
    """

    c = 299_792_458.0 * 1.0e10  # in AA/s
    wave = [100.0, 1.0e5]
    flam = [c * 1.0e-4, 1.0e5]
    fnu = flam_to_fnu(
        flam,
        wave,
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.erg / u.second / u.cm**2 / u.Hz,
    )
    assert np.allclose(fnu, [1.0, 1.0e15 / c])
