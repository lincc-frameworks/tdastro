import numpy as np
from astropy.cosmology import WMAP9, Planck18
from tdastro.astro_utils.redshift import RedshiftDistFunc, redshift_to_distance
from tdastro.sources.basic_sources import StepSource


def test_redshifted_flux_densities() -> None:
    """Test that we correctly calculate redshifted values."""
    times = np.linspace(0, 100, 1000)
    wavelengths = np.array([100.0, 200.0, 300.0])
    t0 = 10.0
    t1 = 30.0
    brightness = 50.0

    for redshift in [0.0, 0.5, 2.0, 3.0, 30.0]:
        model_redshift = StepSource(brightness=brightness, t0=t0, t1=t1, redshift=redshift)
        values_redshift = model_redshift.evaluate(times, wavelengths)

        for i, time in enumerate(times):
            if t0 <= time and time <= (t1 - t0) * (1 + redshift) + t0:
                assert np.all(values_redshift[i] == brightness * (1 + redshift))
            else:
                assert np.all(values_redshift[i] == 0.0)


def test_redshift_to_distance():
    """Test that we can convert the redshift to a distance using a given cosmology."""
    wmap9_val = redshift_to_distance(1100, cosmology=WMAP9)
    planck18_val = redshift_to_distance(1100, cosmology=Planck18)

    assert abs(planck18_val - wmap9_val) > 1000.0
    assert 13.0 * 1e12 < wmap9_val < 16.0 * 1e12
    assert 13.0 * 1e12 < planck18_val < 16.0 * 1e12


def test_redshift_dist_func_node():
    """Test the RedshiftDistFunc node."""
    node = RedshiftDistFunc(redshift=1100, cosmology=Planck18)
    state = node.sample_parameters()
    assert 13.0 * 1e12 < node.get_param(state, "function_node_result") < 16.0 * 1e12
