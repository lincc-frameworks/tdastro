import numpy as np
import pytest
from tdastro.astro_utils.agn_utils import (
    agn_accretion_rate,
    agn_blackhole_accretion_rate,
    agn_bolometric_luminosity,
    eddington_ratio_dist_fun,
)
from tdastro.consts import M_SUN_G


def test_agn_accretion_rate():
    """Test that we can compute the accretion rate from mass of the black hole."""
    # This is a change detection test to make sure the results match previous code.

    # No BH = no accretion.
    assert agn_accretion_rate(0.0) == 0.0

    # For a 1000 solar mass black hole, the accretion rate should be 1.4e21 g/s.
    assert agn_accretion_rate(1000.0 * M_SUN_G) == pytest.approx(1.4e21)

    # Test that the operation is vectorized.
    num_suns = np.array([0.5, 1.0, 10.0, 50.0])
    masses = M_SUN_G * num_suns
    expected = 1.4e18 * num_suns
    assert np.allclose(agn_accretion_rate(masses), expected)


def test_agn_black_hole_accretion_rate():
    """Test that we can compute the black hole's accretion rate."""
    # This is a change detection test to make sure the results match previous code.

    # No accretion => no blackhole accretion.
    assert agn_blackhole_accretion_rate(0.0, 0.5) == 0.0
    assert agn_blackhole_accretion_rate(0.0, 1.0) == 0.0
    assert agn_blackhole_accretion_rate(0.0, 2.0) == 0.0

    # For a total accretion rate of 1.4e21 g/s and different ratios,
    # compute the black hole accretion rate.
    assert agn_blackhole_accretion_rate(1.4e21, 1.0) == pytest.approx(1.4e21)
    assert agn_blackhole_accretion_rate(1.4e21, 0.5) == pytest.approx(7.0e20)
    assert agn_blackhole_accretion_rate(1.4e21, 2.0) == pytest.approx(2.8e21)

    # Test that the operation is vectorized.
    rates = np.array([0.5, 1.0, 10.0, 50.0])
    ratios = np.array([1.0, 1.0, 2.0, 2.0])
    assert np.allclose(agn_blackhole_accretion_rate(rates, ratios), rates * ratios)


def test_agn_bolometric_luminosity():
    """Test that we can compute the bolometric luminosity from mass of the black hole."""
    # This is a change detection test to make sure the results match previous code.

    # No BH = no luminosity.
    assert agn_bolometric_luminosity(1.0, 0.0) == 0.0

    # For a 100 solar mass black hole and 1.0 ratio, the bolometric luminosity should be 1.26e40
    assert agn_bolometric_luminosity(1.0, 100.0 * M_SUN_G) == pytest.approx(1.26e40)

    # For a 100 solar mass black hole and 0.5 ratio, the bolometric luminosity should be 6.3e39
    assert agn_bolometric_luminosity(0.5, 100.0 * M_SUN_G) == pytest.approx(6.3e39)

    # Test that the operation is vectorized.
    num_suns = np.array([0.5, 1.0, 10.0, 50.0])
    ratios = np.array([1.0, 1.0, 2.0, 2.0])
    masses = M_SUN_G * num_suns

    expected = 1.26e38 * num_suns * ratios
    assert np.allclose(agn_bolometric_luminosity(ratios, masses), expected)


def test_eddington_ratio_dist_fun():
    """Test the eddington_ratio_dist_fun function."""
    # This is a change detection test to make sure the results match previous code.

    # Test that we can draw samples and the fall within the expected bounds.
    rng = np.random.default_rng(100)
    for type in ["blue", "red"]:
        for edd_ratio in [0.5, 1.0, 2.0]:
            samples = eddington_ratio_dist_fun(edd_ratio, type, rng, 1000)
            assert len(samples) == 1000
            assert np.all(samples >= 0.0)

    # Test that if we draw a single sample it is a float.
    sample = eddington_ratio_dist_fun(1.0, "blue")
    assert isinstance(sample, float)