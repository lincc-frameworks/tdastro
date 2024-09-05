import numpy as np
from numpy.testing import assert_allclose
from tdastro.astro_utils.noise_model import poisson_flux_std
from tdastro.consts import GAUSS_EFF_AREA2FWHM_SQ


def test_poisson_flux_std_flux():
    """Test poisson_flux_std for photon noise dominated regime"""
    # The results should be independent of some input parameters
    rng = np.random.default_rng(None)

    flux = 10 ** rng.uniform(0.0, 5.0, 100)
    expected_flux_err = np.sqrt(flux)

    flux_err = poisson_flux_std(
        flux=flux,
        pixel_scale=rng.uniform(),
        total_exposure_time=rng.uniform(),
        exposure_count=rng.integers(1, 100),
        footprint=rng.uniform(),
        sky=0.0,
        zp=1.0,
        readout_noise=0.0,
        dark_current=0.0,
    )

    assert_allclose(flux_err, expected_flux_err, rtol=1e-10)


def test_poisson_flux_std_sky():
    """Test poisson_flux_std for sky noise dominated regime"""
    # The results should be independent of some input parameters
    rng = np.random.default_rng(None)

    n = 100

    sky = 10 ** rng.uniform(-2.0, 2.0, n)
    footprint = 10 ** rng.uniform(0.0, 2.0, n)
    expected_flux_err = np.sqrt(sky * footprint)

    flux_err = poisson_flux_std(
        flux=0.0,
        pixel_scale=rng.uniform(),
        total_exposure_time=rng.uniform(),
        exposure_count=rng.integers(1, 100),
        footprint=footprint,
        sky=sky,
        zp=1.0,
        readout_noise=0.0,
        dark_current=0.0,
    )

    assert_allclose(flux_err, expected_flux_err, rtol=1e-10)


def test_poisson_flux_std_readout():
    """Test poisson_flux_std for readout noise dominated regime"""
    # The results should be independent of some input parameters
    rng = np.random.default_rng(None)

    n = 100

    readout_noise = 10 ** rng.uniform(-2.0, 2.0, n)
    pixel_scale = 10 ** rng.uniform(-1.0, 1.0, n)
    footprint = 10 ** rng.uniform(0.0, 2.0, n)
    exposure_count = rng.integers(1, 100, n)
    expected_flux_err = readout_noise * np.sqrt(footprint / pixel_scale**2) * np.sqrt(exposure_count)

    flux_err = poisson_flux_std(
        flux=0.0,
        pixel_scale=pixel_scale,
        total_exposure_time=rng.uniform(),
        exposure_count=exposure_count,
        footprint=footprint,
        sky=0.0,
        zp=1.0,
        readout_noise=readout_noise,
        dark_current=0.0,
    )

    assert_allclose(flux_err, expected_flux_err, rtol=1e-10)


def test_poisson_flux_std_dark():
    """Test poisson_flux_std for dark current noise dominated regime"""
    # The results should be independent of some input parameters
    rng = np.random.default_rng(None)

    n = 100

    dark_current = 10 ** rng.uniform(-2.0, 2.0, n)
    total_exposure_time = rng.uniform(1.0, 3.0, n)
    pixel_scale = 10 ** rng.uniform(-1.0, 1.0, n)
    footprint = 10 ** rng.uniform(0.0, 2.0, n)

    dark_current_total = dark_current * total_exposure_time * footprint / pixel_scale**2
    expected_flux_err = np.sqrt(dark_current_total)

    flux_err = poisson_flux_std(
        flux=0.0,
        pixel_scale=pixel_scale,
        total_exposure_time=total_exposure_time,
        exposure_count=rng.integers(1, 100, n),
        footprint=footprint,
        sky=0.0,
        zp=1.0,
        readout_noise=0.0,
        dark_current=dark_current,
    )

    assert_allclose(flux_err, expected_flux_err, rtol=1e-10)


def test_gauss_eff_area2fwhm_sq():
    """Test GAUSS_EFF_AREA2FWHM_SQ to be close to 2.266

    Find 2.266 here:
    https://smtn-002.lsst.io/v/OPSIM-1171/index.html
    """
    assert_allclose(GAUSS_EFF_AREA2FWHM_SQ, 2.266, atol=1e-3)
