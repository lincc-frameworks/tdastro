import numpy as np
import pytest
from citation_compass import get_all_citations
from tdastro.astro_utils.agn_utils import eddington_ratio_dist_fun


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

    # Test that we fail if we use an incorrect galaxy type.
    with pytest.raises(ValueError):
        eddington_ratio_dist_fun(1.0, "green")


def test_eddington_ratio_dist_fun_citations():
    """Test the citations for the eddington_ratio_dist_fun function."""
    citations = get_all_citations()

    # Confirm we find both citations.
    found1 = False
    found2 = False
    for citation in citations:
        if "Sartori et. al. 2019" in citation:
            found1 = True
        if "Weigel et. al. 2017" in citation:
            found2 = True
    assert found1 and found2
