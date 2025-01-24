import numpy as np
import pytest
from tdastro.astro_utils.agn_utils import (
    agn_accretion_rate,
    agn_bolometric_luminosity,
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
