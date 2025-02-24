import numpy as np
from tdastro.astro_utils.dustmap import ConstantHemisphereDustMap, DustmapWrapper
from tdastro.math_nodes.given_sampler import GivenValueList


def test_constant_hemisphere_dust_map():
    """Test that we can create and directly query a ConstantHemisphereDustMap object."""
    dust_map = ConstantHemisphereDustMap(north_ebv=0.5, south_ebv=0.4)
    assert dust_map.north_ebv == 0.5
    assert dust_map.south_ebv == 0.4

    # Check that the dust map returns the correct values for different locations.
    assert dust_map.compute_ebv(0.0, 45.0) == 0.5
    assert dust_map.compute_ebv(-10.0, 70.0) == 0.5
    assert dust_map.compute_ebv(5.0, 0.001) == 0.5
    assert dust_map.compute_ebv(0.0, -45.0) == 0.4
    assert dust_map.compute_ebv(20.0, -0.001) == 0.4

    ebvs = dust_map.compute_ebv(
        [0.0, 0.0, 20.0, 340.0, 250.0, 180.0],
        [45.0, -45.0, 15.0, -70.0, -90.0, 0.0],
    )
    assert np.array_equal(ebvs, [0.5, 0.4, 0.5, 0.4, 0.4, 0.5])


def test_dust_map_wrapper():
    """Test that we can create and sample a wrapped dust map."""
    dec_vals = np.arange(-90.0, 90.0, 15.0)
    ra_vals = np.arange(0.0, 360.0, 30.0)
    fake_dust_map = ConstantHemisphereDustMap(north_ebv=0.8, south_ebv=0.1)

    dust_node1 = DustmapWrapper(
        dust_map=fake_dust_map,
        ra=GivenValueList(ra_vals),
        dec=GivenValueList(dec_vals),
        node_label="dust",
    )
    samples = dust_node1.sample_parameters(num_samples=len(dec_vals))
    assert np.all(samples["dust"]["ebv"][dec_vals >= 0] == 0.8)
    assert np.all(samples["dust"]["ebv"][dec_vals < 0] == 0.1)

    # Try adding an adjustment function to the dust map.
    def _ebv_adjust(value):
        return value + 0.1

    dust_node2 = DustmapWrapper(
        dust_map=fake_dust_map,
        ebv_func=_ebv_adjust,
        ra=GivenValueList(ra_vals),
        dec=GivenValueList(dec_vals),
        node_label="dust",
    )
    samples = dust_node2.sample_parameters(num_samples=len(dec_vals))
    assert np.all(samples["dust"]["ebv"][dec_vals >= 0] == 0.9)
    assert np.all(samples["dust"]["ebv"][dec_vals < 0] == 0.2)
