import numpy as np
from scipy.optimize import fsolve
from tdastro.opsim.ztf_opsim import ZTFOpsim, calculate_ztf_zero_points, create_random_ztf_opsim


def fluxeq(flux, sky=None, gain=None, readnoise=None, nexposure=1, fwhm=None, darkcurrent=None, exptime=None):
    """Define the equation to solve for flux at 5-sigma limit"""

    npix = 2.266 * fwhm**2  # =4*pi*sigma**2 = pi/2/ln2 * FWHM**2
    y = (
        flux**2
        - 25 * flux
        - 25
        * (sky * npix * gain + readnoise**2 * nexposure * npix + darkcurrent * npix * exptime * nexposure)
    )
    return y


def test_calculate_ztf_zero_points():
    """Test the calculation with numerical solution to the maglim function."""

    maglim = 19.0
    sky = 100.0
    fwhm = 1.5
    gain = 1.0
    readnoise = 10.0
    darkcurrent = 0.1
    exptime = 30.0

    root = fsolve(
        lambda x: fluxeq(
            x, sky=sky, gain=gain, readnoise=readnoise, fwhm=fwhm, darkcurrent=darkcurrent, exptime=exptime
        ),
        [100],
    )
    zp_expected = 2.5 * np.log10(root / gain) + maglim

    zp_cal = calculate_ztf_zero_points(
        maglim=maglim,
        sky=sky,
        fwhm=fwhm,
        gain=gain,
        readnoise=readnoise,
        darkcurrent=darkcurrent,
        exptime=exptime,
        nexposure=1,
    )

    assert np.isclose(zp_cal, zp_expected)


def test_ztf_opsim_init():
    """Test initializing ZTFOpsim."""
    opsim_table = create_random_ztf_opsim(100).table
    opsim = ZTFOpsim(table=opsim_table)

    assert opsim.has_columns(["zp", "obsmjd"])
