"""Test the utility functions for computing AGN parameters. These are mostly change
detection tests to make sure the results match the code from SNANA:
https://github.com/RickKessler/SNANA/blob/master/src/gensed_AGN.py
"""

import numpy as np
import pytest
from tdastro.consts import M_SUN_G
from tdastro.math_nodes.np_random import NumpyRandomFunc
from tdastro.sources.gensed_agn import AGN


def test_agn_match_size_to_lambda():
    """Test that we can match different types and shapes of data."""
    # Scalar by scalar.
    assert np.isscalar(AGN.match_size_to_lambda(1.0, 1.0))

    # Scalar by wavelength array.
    result = AGN.match_size_to_lambda(np.array([1.0, 2.0, 3.0]), 0.5)
    assert result.shape == (3,)
    assert np.all(result == 0.5)

    # A bunch of samples but a scalar wavelength should produce an error.
    with pytest.raises(ValueError):
        _ = AGN.match_size_to_lambda(2.0, np.array([0.5, 1.0, 1.5]))

    # A bunch of samples but a flat wavelength array should produce a rectangular array.
    result = AGN.match_size_to_lambda(np.array([2.0, 3.0]), np.array([0.5, 1.0, 2.0]))
    assert result.shape == (3, 2)
    assert np.array_equal(result[:, 0], np.array([0.5, 1.0, 2.0]))
    assert np.array_equal(result[:, 1], np.array([0.5, 1.0, 2.0]))

    # A bunch of samples and a 2D wavelength array.
    result = AGN.match_size_to_lambda(np.array([[2.0, 3.0], [4.0, 5.0]]), np.array([0.5, 1.0]))
    assert np.array_equal(result[:, 0], np.array([0.5, 1.0]))
    assert np.array_equal(result[:, 1], np.array([0.5, 1.0]))

    # We have an error if the number of samples is different.
    with pytest.raises(ValueError):
        _ = AGN.match_size_to_lambda(np.array([[2.0, 3.0], [4.0, 5.0]]), np.array([0.5, 1.0, 1.5]))


def test_agn_accretion_rate():
    """Test that we can compute the accretion rate from mass of the black hole."""
    # This is a change detection test to make sure the results match previous code.

    # No BH = no accretion.
    assert AGN.compute_accretion_rate(0.0) == 0.0

    # For a 1000 solar mass black hole, the accretion rate should be 1.4e21 g/s.
    assert AGN.compute_accretion_rate(1000.0 * M_SUN_G) == pytest.approx(1.4e21)

    # Test that the operation is vectorized.
    num_suns = np.array([0.5, 1.0, 10.0, 50.0])
    masses = M_SUN_G * num_suns
    expected = 1.4e18 * num_suns
    assert np.allclose(AGN.compute_accretion_rate(masses), expected)


def test_agn_black_hole_accretion_rate():
    """Test that we can compute the black hole's accretion rate."""
    # This is a change detection test to make sure the results match previous code.

    # No accretion => no blackhole accretion.
    assert AGN.compute_blackhole_accretion_rate(0.0, 0.5) == 0.0
    assert AGN.compute_blackhole_accretion_rate(0.0, 1.0) == 0.0
    assert AGN.compute_blackhole_accretion_rate(0.0, 2.0) == 0.0

    # For a total accretion rate of 1.4e21 g/s and different ratios,
    # compute the black hole accretion rate.
    assert AGN.compute_blackhole_accretion_rate(1.4e21, 1.0) == pytest.approx(1.4e21)
    assert AGN.compute_blackhole_accretion_rate(1.4e21, 0.5) == pytest.approx(7.0e20)
    assert AGN.compute_blackhole_accretion_rate(1.4e21, 2.0) == pytest.approx(2.8e21)

    # Test that the operation is vectorized.
    rates = np.array([0.5, 1.0, 10.0, 50.0])
    ratios = np.array([1.0, 1.0, 2.0, 2.0])
    assert np.allclose(AGN.compute_blackhole_accretion_rate(rates, ratios), rates * ratios)


def test_agn_bolometric_luminosity():
    """Test that we can compute the bolometric luminosity from mass of the black hole."""
    # This is a change detection test to make sure the results match previous code.

    # No BH = no luminosity.
    assert AGN.compute_bolometric_luminosity(1.0, 0.0) == 0.0

    # For a 100 solar mass black hole and 1.0 ratio, the bolometric luminosity should be 1.26e40
    assert AGN.compute_bolometric_luminosity(1.0, 100.0 * M_SUN_G) == pytest.approx(1.26e40)

    # For a 100 solar mass black hole and 0.5 ratio, the bolometric luminosity should be 6.3e39
    assert AGN.compute_bolometric_luminosity(0.5, 100.0 * M_SUN_G) == pytest.approx(6.3e39)

    # Test that the operation is vectorized.
    num_suns = np.array([0.5, 1.0, 10.0, 50.0])
    ratios = np.array([1.0, 1.0, 2.0, 2.0])
    masses = M_SUN_G * num_suns

    expected = 1.26e38 * num_suns * ratios
    assert np.allclose(AGN.compute_bolometric_luminosity(ratios, masses), expected)


def test_agn_compute_mag_i():
    """Test that we can compute the i band magnitude from the bolometric luminosity."""
    # This is a change detection test to make sure the results match previous code.

    assert AGN.compute_mag_i(1.0) == pytest.approx(90.0)

    # Test that the operation is vectorized.
    l_bol = np.array([0.5, 1.0, 10.0, 50.0])
    expected = np.array([90.75257499, 90.0, 87.5, 85.75257499])
    assert np.allclose(AGN.compute_mag_i(l_bol), expected)


def test_agn_compute_r_0():
    """Test that we can compute an AGN's r_0 from its r_in."""
    # This is a change detection test to make sure the results match previous code.

    assert AGN.compute_r_0(1.0) == pytest.approx(1.361111111111111)

    # Test that the operation is vectorized.
    r_in = np.array([0.5, 1.0, 10.0, 50.0])
    expected = 1.361111111111111 * r_in
    assert np.allclose(AGN.compute_r_0(r_in), expected)


def test_agn_compute_temp_at_r_0():
    """Test that we can compute the effective temperature at r0."""
    # This is a change detection test to make sure the results match previous code.

    assert AGN.compute_temp_at_r_0(1000.0, 100.0, 5.0) == pytest.approx(0.28248581706023856)


def test_agn_compute_x_fun():
    """Test that we can compute the variable of integration x."""
    # This is a change detection test to make sure the results match previous code.

    assert AGN.compute_x_fun(1000.0, 100.0, 5.0, 2.0) == pytest.approx(9.541743841215435e-10)


def test_agn_structure_function_at_inf():
    """Test that we can compute the structure function at infinity."""
    # This is a change detection test to make sure the results match previous code.
    assert AGN.structure_function_at_inf(1.0) == pytest.approx(0.002417612741449265)

    # Test a range of parameters.
    lam = np.array([10.0, 20.0])
    mag_i = np.array([-23.0, -22.5, -23.5, -23.0])
    blackhole_mass = 1e9 * M_SUN_G * np.array([0.5, 1.0, 1.5, 2.0])
    expected = np.array(
        [
            [0.00070827, 0.00050817],
            [0.00093194, 0.00066864],
            [0.00074316, 0.0005332],
            [0.00090902, 0.000652],
        ],
    )
    assert np.allclose(AGN.structure_function_at_inf(lam, mag_i, blackhole_mass), expected, atol=1e-5)


def test_agn_tau_v_drw():
    """Test that we can compute the timescale (tau_v) for the DRW model."""
    # This is a change detection test to make sure the results match previous code.
    assert AGN.tau_v_drw(1.0) == pytest.approx(1404.9141979667831)

    # Test a range of parameters. Because we are using a 1-d array of wavelengths
    # and a 1-d array of samples, we expect the results to be a 2-d array.
    lam = np.array([10.0, 20.0])
    mag_i = np.array([-23.0, -22.5, -23.5, -23.0])
    blackhole_mass = 1e9 * M_SUN_G * np.array([0.5, 1.0, 1.5, 2.0])
    expected = np.array(
        [
            [1796.52598149, 2021.19679844],
            [2151.04651325, 2420.05313069],
            [2185.89333503, 2459.2578432],
            [2403.62161353, 2704.21489028],
        ],
    )
    assert np.allclose(AGN.tau_v_drw(lam, mag_i, blackhole_mass), expected)


def test_eddington_ratio_dist_fun():
    """Test the eddington_ratio_dist_fun function."""
    # This is a change detection test to make sure the results match previous code.

    # Test that we can draw samples and the fall within the expected bounds.
    rng = np.random.default_rng(100)
    for type in ["blue", "red"]:
        for edd_ratio in [0.5, 1.0, 2.0]:
            samples = AGN.eddington_ratio_dist_fun(edd_ratio, type, rng, 1000)
            assert len(samples) == 1000
            assert np.all(samples >= 0.0)

    # Test that if we draw a single sample it is a float.
    sample = AGN.eddington_ratio_dist_fun(1.0, "blue")
    assert isinstance(sample, float)


def test_create_agn():
    """Test that we can create an AGN object and the derived parameters are correct."""
    # Select the black hole mass uniformly between 1e9 and 2e9 solar masses.
    bh_mass_sampler = NumpyRandomFunc("uniform", low=1e9 * M_SUN_G, high=2e9 * M_SUN_G, seed=100)
    agn_node = AGN(
        t0=0.0,
        blackhole_mass=bh_mass_sampler,
        lam=np.array([1000.0, 2000.0, 3000.0, 4000.0]),
        edd_ratio=0.9,
        node_label="AGN",
    )
    state = agn_node.sample_parameters(num_samples=10_000)

    # Check that the parameters are within the expected ranges.
    bh_masses = state["AGN"]["blackhole_mass"]
    assert np.all(bh_masses >= 1e9 * M_SUN_G)
    assert np.all(bh_masses <= 2e9 * M_SUN_G)
    assert np.unique(bh_masses).size > 1_000

    assert np.allclose(state["AGN"]["accretion_rate"], 1.4e18 * bh_masses / M_SUN_G)
    assert np.allclose(state["AGN"]["blackhole_accretion_rate"], 0.9 * state["AGN"]["accretion_rate"])
    assert np.allclose(
        state["AGN"]["bolometric_luminosity"], AGN.compute_bolometric_luminosity(0.9, bh_masses)
    )
    assert np.allclose(state["AGN"]["mag_i"], AGN.compute_mag_i(state["AGN"]["bolometric_luminosity"]))
