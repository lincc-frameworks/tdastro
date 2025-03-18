import numpy as np
from tdastro.astro_utils.mag_flux import flux2mag
from tdastro.astro_utils.zeropoint import flux_electron_zeropoint, magnitude_electron_zeropoint
from tdastro.opsim.opsim import LSSTCAM_PIXEL_SCALE, _lsstcam_dark_current, _lsstcam_readout_noise


def test_magnitude_electron_zeropoint():
    """Test that instrumental zeropoints are correct"""
    # Reproducing magnitude corresponding to S/N=5 from
    # https://smtn-002.lsst.io/v/OPSIM-1171/index.html
    fwhm_eff = {
        "u": 0.92,
        "g": 0.87,
        "r": 0.83,
        "i": 0.80,
        "z": 0.78,
        "y": 0.76,
    }
    fwhm_eff_getter = np.vectorize(fwhm_eff.get)
    sky_brightness = {
        "u": 23.05,
        "g": 22.25,
        "r": 21.20,
        "i": 20.46,
        "z": 19.61,
        "y": 18.60,
    }
    sky_brightness_getter = np.vectorize(sky_brightness.get)

    bands = list(fwhm_eff.keys())
    exptime = 30
    airmass = 1
    s2n = 5
    zp = magnitude_electron_zeropoint(
        band=bands, airmass=airmass, exptime=exptime, instr_zp=None, ext_coeff=None
    )
    sky_count_per_arcsec_sq = np.power(10.0, -0.4 * (sky_brightness_getter(bands) - zp))
    readout_per_arcsec_sq = _lsstcam_readout_noise**2 / LSSTCAM_PIXEL_SCALE**2
    dark_per_arcsec_sq = _lsstcam_dark_current * exptime / LSSTCAM_PIXEL_SCALE**2
    count_per_arcsec_sq = sky_count_per_arcsec_sq + readout_per_arcsec_sq + dark_per_arcsec_sq

    area = 2.266 * fwhm_eff_getter(bands) ** 2  # effective seeing area in arcsec^2
    n_background = count_per_arcsec_sq * area
    # sky-dominated regime would be n_signal = s2n * np.sqrt(n_background)
    n_signal = 0.5 * (s2n**2 + np.sqrt(s2n**4 + 4 * s2n**2 * n_background))
    mag_signal = zp - 2.5 * np.log10(n_signal)

    m5_desired = {
        "u": 23.70,
        "g": 24.97,
        "r": 24.52,
        "i": 24.13,
        "z": 23.56,
        "y": 22.55,
    }
    assert list(m5_desired) == bands
    m5_desired_getter = np.vectorize(m5_desired.get)

    np.testing.assert_allclose(mag_signal, m5_desired_getter(bands), atol=0.1)


def test_magnitude_electron_zeropoint_docstring():
    """Check if magnitude_electron_zeropoint has a docstring"""
    assert magnitude_electron_zeropoint.__doc__ is not None
    assert len(magnitude_electron_zeropoint.__doc__) > 100


def test_flux_electron_zeropoint():
    """Test that bandflux zeropoints are correct"""
    # Here we just check that magnitude-flux conversion is correct
    airmass = np.array([1, 1.5, 2]).reshape(-1, 1, 1)
    exptime = np.array([30, 38, 45]).reshape(1, -1, 1)
    bands = ["u", "g", "r", "i", "z", "y"]
    mag = magnitude_electron_zeropoint(
        band=bands, airmass=airmass, exptime=exptime, instr_zp=None, ext_coeff=None
    )
    flux = flux_electron_zeropoint(
        band=bands, airmass=airmass, exptime=exptime, instr_zp_mag=None, ext_coeff=None
    )
    np.testing.assert_allclose(mag, flux2mag(flux), rtol=1e-10)


def test_flux_electron_zeropoint_docstring():
    """Check if flux_electron_zeropoint has a docstring"""
    assert flux_electron_zeropoint.__doc__ is not None
    assert len(flux_electron_zeropoint.__doc__) > 100
