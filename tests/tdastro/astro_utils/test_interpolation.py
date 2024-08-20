import numpy as np
from tdastro.astro_utils.interpolation import grids_have_same_step


def test_check_grids_same_step():
    """Test check_grids_same_step_size function."""
    a1 = np.array([1, 2, 3])
    a2 = np.array([11, 12, 13, 14, 15])
    a3 = np.array([1, 2, 3, 40])
    assert grids_have_same_step(a1, a2)
    assert not grids_have_same_step(a1, a3)
