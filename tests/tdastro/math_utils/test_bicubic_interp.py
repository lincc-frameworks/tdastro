import jax.numpy as jnp
import numpy as np
import pytest

from tdastro.math_utils.bicubic_interp import _kernel_value, _kernval, BicubicInterpolator

def test_bicubic_interpolator():
    """Test that we can create and query a bicubic interpolator."""
    x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    z_vals = np.full((len(x_vals), len(y_vals)), 1.0)
    interp_obj = BicubicInterpolator(x_vals, y_vals, z_vals)

    interp_obj(
        [-0.5, 1.0, 1.5, 2.1, 3.0, 4.0, 5.5],
        [-0.5, 0.0, 0.5, 1.1, 1.5, 2.0, 3.5],
    )
    assert False


def test_kernel_values():
    """Test that we can evaluate the kernel."""
    input = jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    all_results = _kernel_value(input)

    for i, val in enumerate(input):
        current = _kernval(val)
        assert current == all_results[i]


def test_bad_bicubic_interpolator():
    """Test that we catch failure cases for creating a bicubic interpolator."""
    _ = BicubicInterpolator([1.0], [1.0], [[1.0]])

    # No empty x or y
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([], [1.0], [[1.0]])
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([1.0], [], [[1.0]])

    # x and y must each be 1-d and z must be 2-d.
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([[1.0]], [1.0], [[1.0]])
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([1.0], [[1.0]], [[1.0]])
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([1.0], [1.0], [1.0])
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([1.0], [1.0], [[[1.0]]])

    # The x and y values must be in sorted order.
    z = np.full((3, 4), 1.0)
    _ = BicubicInterpolator([1.0, 2.0, 3.0], [0.0, 0.5, 1.0, 1.5], z)
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([2.0, 1.0, 3.0], [0.0, 0.5, 1.0, 1.5], z)
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([1.0, 2.0, 3.0], [0.0, 2.5, 1.0, 1.5], z)

    # Sizes must match z.
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([1.0, 2.0, 3.0, 4.0], [0.0, 2.5, 1.0, 1.5], z)
    with pytest.raises(ValueError):
        _ = BicubicInterpolator([1.0, 2.0, 3.0], [0.0, 2.5, 1.0], z)
