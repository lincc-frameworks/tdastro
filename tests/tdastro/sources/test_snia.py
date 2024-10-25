from tdastro.example_runs.simulate_snia import run_snia_end2end


def test_snia_end2end(oversampled_observations, passbands_dir):
    """Test that the end to end run works."""
    num_samples = 1
    res_list, passbands = run_snia_end2end(
        oversampled_observations,
        passbands_dir,
        nsample=num_samples,
        check_sncosmo=True,
    )
    assert len(res_list) == num_samples
    assert len(passbands) == 2
