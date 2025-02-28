import numpy as np
from tdastro.math_nodes.ra_dec_sampler import (
    OpSimRADECSampler,
    OpSimUniformRADECSampler,
    UniformRADEC,
)
from tdastro.opsim.opsim import OpSim


def test_uniform_ra_dec():
    """Test that we can generate numbers from a uniform distribution on a sphere."""
    sampler_node = UniformRADEC(seed=100, node_label="sampler")

    # Test we can generate a single value.
    (ra, dec) = sampler_node.generate(num_samples=1)
    assert 0.0 <= ra <= 360.0
    assert -90.0 <= dec <= 90.0

    # Generate many samples.
    num_samples = 20_000
    state = sampler_node.sample_parameters(num_samples=num_samples)

    all_ra = state["sampler"]["ra"]
    assert len(all_ra) == num_samples
    assert np.all(all_ra >= 0.0)
    assert np.all(all_ra <= 360.0)

    all_dec = state["sampler"]["dec"]
    assert len(all_dec) == num_samples
    assert np.all(all_dec >= -90.0)
    assert np.all(all_dec <= 90.0)

    # Compute histograms of RA and dec values.
    ra_bins = np.zeros(36)
    dec_bins = np.zeros(18)
    for idx in range(num_samples):
        ra_bins[int(all_ra[idx] / 10.0)] += 1
        dec_bins[int((all_dec[idx] + 90.0) / 10.0)] += 1

    # Check that all RA bins have approximately equal samples.
    expected_count = num_samples / 36
    for bin_count in ra_bins:
        assert 0.8 <= bin_count / expected_count <= 1.2

    # Check that the dec bins around the poles have less samples
    # than the bins around the equator.
    assert dec_bins[0] < 0.25 * dec_bins[9]
    assert dec_bins[17] < 0.25 * dec_bins[10]

    # Check that we can generate uniform samples in radians.
    sampler_node2 = UniformRADEC(seed=100, node_label="sampler2", use_degrees=False)
    state2 = sampler_node2.sample_parameters(num_samples=num_samples)

    all_ra = state2["sampler2"]["ra"]
    assert len(all_ra) == num_samples
    assert np.all(all_ra >= 0.0)
    assert np.all(all_ra <= 2.0 * np.pi)

    all_dec = state2["sampler2"]["dec"]
    assert len(all_dec) == num_samples
    assert np.all(all_dec >= -np.pi)
    assert np.all(all_dec <= np.pi)


def test_opsim_ra_dec_sampler():
    """Test that we can sample from am OpSim object."""
    values = {
        "observationStartMJD": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "fieldRA": np.array([15.0, 30.0, 15.0, 0.0, 60.0]),
        "fieldDec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        "zp_nJy": np.ones(5),
    }
    ops_data = OpSim(values)
    assert len(ops_data) == 5

    sampler_node = OpSimRADECSampler(ops_data, in_order=True)

    # Test we can generate a single value.
    (ra, dec, time) = sampler_node.generate(num_samples=1)
    assert ra == 15.0
    assert dec == -10.0
    assert time == 0.0

    # Test we can generate multiple observations
    (ra, dec, time) = sampler_node.generate(num_samples=2)
    assert np.allclose(ra, [30.0, 15.0])
    assert np.allclose(dec, [-5.0, 0.0])
    assert np.allclose(time, [1.0, 2.0])

    # Do randomized sampling.
    sampler_node2 = OpSimRADECSampler(ops_data, in_order=False, seed=100, node_label="sampler")
    state = sampler_node2.sample_parameters(num_samples=5000)

    # Check that the samples are uniform and consistent.
    int_times = state["sampler"]["time"].astype(int)
    assert np.allclose(state["sampler"]["ra"], values["fieldRA"][int_times])
    assert np.allclose(state["sampler"]["dec"], values["fieldDec"][int_times])
    assert len(int_times[int_times == 0]) > 750
    assert len(int_times[int_times == 1]) > 750
    assert len(int_times[int_times == 2]) > 750
    assert len(int_times[int_times == 3]) > 750
    assert len(int_times[int_times == 4]) > 750

    # Do randomized sampling with offsets.
    sampler_node3 = OpSimRADECSampler(ops_data, in_order=False, seed=100, radius=0.1, node_label="sampler")
    state = sampler_node3.sample_parameters(num_samples=5000)

    # Check that the samples are not all the centers (unique values > 5) but are close.
    int_times = state["sampler"]["time"].astype(int)
    assert len(np.unique(state["sampler"]["ra"])) > 5
    assert len(np.unique(state["sampler"]["dec"])) > 5
    assert np.allclose(state["sampler"]["ra"], values["fieldRA"][int_times], atol=0.2)
    assert np.allclose(state["sampler"]["dec"], values["fieldDec"][int_times], atol=0.2)


def test_opsim_uniform_ra_dec_sampler():
    """Test that we can sample uniformly from am OpSim object."""
    # Create an opsim with two points in different hemispheres.
    values = {
        "observationStartMJD": np.array([0.0, 1.0]),
        "fieldRA": np.array([15.0, 195.0]),
        "fieldDec": np.array([75.0, -75.0]),
        "zp_nJy": np.ones(2),
    }
    ops_data = OpSim(values)
    assert len(ops_data) == 2

    # Use a very large radius so we do not reject too many samples.
    sampler_node = OpSimUniformRADECSampler(ops_data, radius=70.0, seed=100, node_label="sampler")

    # Test we can generate a single value.
    ra, dec = sampler_node.generate(num_samples=1)
    assert ops_data.is_observed(ra, dec, radius=70.0)

    # Test we can generate many observations
    num_samples = 10_000
    ra, dec = sampler_node.generate(num_samples=num_samples)
    assert np.all(ops_data.is_observed(ra, dec, radius=70.0))

    # We should sample roughly uniformly from the two regions.
    northern_mask = dec > 0.0
    assert np.sum(northern_mask) > 0.4 * num_samples
    assert np.sum(northern_mask) < 0.6 * num_samples
