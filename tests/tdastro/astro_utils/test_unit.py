from astropy import units as u
from tdastro.astro_utils.unit_utils import flam_to_fnu
import numpy as np

def test_flam_to_fnu():
    """Test flam to fnu conversion
    fnu = flam * lam**2 / c
    """

    c_AA = 299_792_458.0*1.e10 #in AA/s
    wave = [100.0, 1.e5]
    flam = [c_AA*1.e-4, 1.e5]
    fnu = flam_to_fnu(flam, wave, wave_unit=u.AA, flam_unit= u.erg / u.second / u.cm**2 / u.AA, fnu_unit=u.erg / u.second / u.cm**2 / u.Hz)
    assert np.allclose(fnu,[1.,1.e15/c_AA])