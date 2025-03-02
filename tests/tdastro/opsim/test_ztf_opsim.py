import numpy as np
import pandas as pd
import pytest
from scipy.optimize import fsolve
from tdastro.opsim.ztf_opsim import (
    ZTFOpsim,
    calculate_ztf_zero_points,
    create_random_ztf_opsim,
)


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
    zp_expected = 2.5 * np.log10(root) + maglim

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

    assert opsim.has_columns(["zp_nJy", "obsmjd"])

    # We have all the attributes set at their default values.
    assert opsim.dark_current == 0.0
    assert opsim.gain == 6.2
    assert opsim.pixel_scale == 1.01
    assert opsim.radius == 2.735
    assert opsim.read_noise == 8


def test_create_ztf_opsim_override():
    """Test that we can override the default survey values."""
    opsim_table = create_random_ztf_opsim(100).table

    opsim = ZTFOpsim(
        table=opsim_table,
        dark_current=0.1,
        gain=7.1,
        pixel_scale=0.1,
        radius=1.0,
        read_noise=5.0,
    )

    # We have all the attributes set at their default values.
    assert opsim.dark_current == 0.1
    assert opsim.gain == 7.1
    assert opsim.pixel_scale == 0.1
    assert opsim.radius == 1.0
    assert opsim.read_noise == 5.0


def test_create_ztf_opsim_no_zp():
    """Create an opsim without a zeropoint column."""
    dates = [
        "2020-01-01 12:00:00.000",
        "2020-01-02 12:00:00.000",
        "2020-01-03 12:00:00.000",
        "2020-01-04 12:00:00.000",
        "2020-01-05 12:00:00.000",
    ]
    values = {
        "obsdate": dates,
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
    }

    # We fail if we do not have the other columns needed:
    # "maglim", "sky", "fwhm", "exptime"
    with pytest.raises(ValueError):
        _ = ZTFOpsim(values)

    values["exptime"] = 0.005 * np.ones(5)
    values["maglim"] = 20.0 * np.ones(5)
    values["scibckgnd"] = np.ones(5)
    values["fwhm"] = 2.3 * np.ones(5)
    opsim = ZTFOpsim(values)

    assert opsim.has_columns("zp_nJy")
    assert np.all(opsim["zp_nJy"] >= 0.0)


def test_noise_calculation():
    """Test that the noise calculation is in the right range."""
    mag = np.array([19.0])
    expected_magerr = np.array([0.1])

    flux_nJy = np.power(10.0, -0.4 * (mag - 31.4))
    opsim = ZTFOpsim(
        table=pd.DataFrame(
            {
                "ra": 0.0,
                "dec": 0.0,
                "scibckgnd": 200.0,
                "maglim": 20.0,
                "fwhm": 2.3,
                "exptime": 30.0,
                "obsdate": "2020-01-01 12:00:00.000",
            },
            index=[0],
        )
    )
    fluxerr_nJy = opsim.bandflux_error_point_source(flux_nJy, 0)
    magerr = 1.086 * fluxerr_nJy / flux_nJy

    np.testing.assert_allclose(magerr, expected_magerr, rtol=0.2)
