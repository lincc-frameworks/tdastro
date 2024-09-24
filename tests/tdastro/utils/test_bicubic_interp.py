import jax.numpy as jnp
import numpy as np
import pytest

from sncosmo.salt2utils import BicubicInterpolator as SNInterp

from tdastro.utils.bicubic_interp import (
    find_indices,
    _kernel_value,
    BicubicInterpolator,
)


def test_bicubic_interpolator():
    """Test that we can create and query a bicubic interpolator."""
    x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    z_vals = np.full((len(x_vals), len(y_vals)), 1.0)

    # Define the query points.
    x_q = jnp.asarray([2.1])
    y_q = jnp.asarray([1.1])

    #x_q = jnp.asarray([-0.5, 1.0, 1.5, 2.1, 3.0, 4.0, 5.5])
    #y_q = jnp.asarray([-0.5, 0.0, 0.5, 1.1, 1.5, 2.0, 3.5])

    # Do the sncosmo interpolation.
    sn_bci = SNInterp(x_vals, y_vals, z_vals)
    res_sn = sn_bci(x_q, y_q)
    print(type(res_sn))
    print(res_sn)

    # Do the JAX interpolation.
    interp_obj = BicubicInterpolator(x_vals, y_vals, z_vals)
    res_jax = interp_obj(x_q, y_q)

    print(type(res_jax))
    print(res_jax)

    assert jnp.allclose(res_sn, res_jax)
    assert False


def test_index_find():
    """Test the function that finds the inserted location."""
    xvals = jnp.asarray([0.1 * i for i in range(100)])
    n_x = len(xvals)

    x_q = jnp.asarray([0.04 * i - 0.1 for i in range(400)])
    ix = find_indices(xvals, x_q)

    assert jnp.all((ix == n_x - 2) | (x_q < xvals[ix + 1]))
    assert jnp.all((ix == 0) | (xvals[ix] <= x_q))


def test_kernel_values():
    """Test that we can evaluate the kernel with the JAX function
    and get the same results as with the original function."""
    input = jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    expected = jnp.asarray([1.0, 0.5625, 0.0, -0.0625, 0.0, 0.0])
    all_results = _kernel_value(input)
    assert jnp.allclose(expected, all_results)


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
