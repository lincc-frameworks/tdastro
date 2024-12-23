import numpy as np
from tdastro.math_nodes.ra_dec_sampler import UniformRADEC


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
