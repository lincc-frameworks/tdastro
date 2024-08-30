import numpy as np
from tdastro.astro_utils.snia_utils import HostmassX1Func
from tdastro.util_nodes.np_random import NumpyRandomFunc


def test_sample_hostmass_x1c():
    """Test that we can sample correctly from HostmassX1Func."""
    num_samples = 5

    hm_node1 = HostmassX1Func(
        hostmass=NumpyRandomFunc("uniform", low=7, high=12, seed=100),
        seed=101,
    )
    states1 = hm_node1.sample_parameters(num_samples=num_samples)
    values1 = hm_node1.get_param(states1, "function_node_result")
    assert len(values1) == num_samples
    assert len(np.unique(values1)) == num_samples

    # If we create a new node with the same hostmas and the same seeds, we get the
    # same results and the same hostmasses.
    hm_node2 = HostmassX1Func(
        hostmass=NumpyRandomFunc("uniform", low=7, high=12, seed=100),
        seed=101,
    )
    states2 = hm_node2.sample_parameters(num_samples=num_samples)
    values2 = hm_node2.get_param(states2, "function_node_result")
    assert np.allclose(values1, values2)
    assert np.allclose(
        hm_node1.get_param(states1, "hostmass"),
        hm_node2.get_param(states2, "hostmass"),
    )

    # If we use a different seed for the hostmass function only, we get
    # different results (i.e. the hostmass parameter is being resampled).
    hm_node3 = HostmassX1Func(
        hostmass=NumpyRandomFunc("uniform", low=7, high=12, seed=102),
        seed=101,
    )
    states3 = hm_node3.sample_parameters(num_samples=num_samples)
    values3 = hm_node3.get_param(states3, "function_node_result")
    assert not np.allclose(values1, values3)
    assert not np.allclose(
        hm_node1.get_param(states1, "hostmass"),
        hm_node3.get_param(states3, "hostmass"),
    )
