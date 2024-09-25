import os

import jax.numpy as jnp
import numpy as np
import pytest
from tdastro.utils.bicubic_interp import (
    BicubicInterpolator,
    _kernel_value,
    find_indices,
)


def test_bicubic_interpolator_simple():
    """Test that we can create and query a bicubic interpolator."""
    x_vals = np.arange(1.0, 5.0, 1.0)
    y_vals = np.arange(0.0, 3.0, 0.5)
    z_vals = np.full((len(x_vals), len(y_vals)), 1.0)
    interp_obj = BicubicInterpolator(x_vals, y_vals, z_vals)

    x_queries = np.arange(0.5, 5.5, 0.5)
    y_queries = np.arange(-0.5, 3.5, 0.5)
    results = interp_obj(x_queries, y_queries)

    # Everything within the grid should be 1.0 and everything
    # outside the grid should be 0.0.
    for i, x_q in enumerate(x_queries):
        for j, y_q in enumerate(y_queries):
            if (x_q < 1.0) or (x_q > 4.0) or (y_q < 0.0) or (y_q > 2.5):
                assert np.abs(results[i, j]) < 1e-8
            else:
                assert np.abs(results[i, j] - 1.0) < 1e-8


def test_bicubic_interpolator():
    """Test that we can create and query a bicubic interpolator."""
    x_vals = np.arange(1.0, 5.0, 1.0)
    num_x = len(x_vals)

    y_vals = np.arange(-1.0, 2.0, 0.5)
    num_y = len(y_vals)

    # Make a wavey plane of values.
    z_vals = np.empty((num_x, num_y))
    for i in range(num_x):
        for j in range(num_y):
            if j % 2 == 0:
                z_vals[i, j] = i + 0.2 * j
            else:
                z_vals[i, j] = i - 0.1 * j

    # Pick some query points.
    x_q = jnp.asarray([-0.5, 1.0, 1.5, 2.0, 2.1, 3.4, 4.0, 5.5])
    y_q = jnp.asarray([-1.5, 0.0, 0.5, 1.1, 2.0, 3.5])

    # Expected values (as computed by the sncosmo version)
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.4, -0.3, 0.54, 0.0, 0.0],
            [0.0, 0.9, 0.2, 1.04, 0.0, 0.0],
            [0.0, 1.4, 0.7, 1.54, 0.0, 0.0],
            [0.0, 1.5, 0.8, 1.64, 0.0, 0.0],
            [0.0, 2.8, 2.1, 2.94, 0.0, 0.0],
            [0.0, 3.4, 2.7, 3.54, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    # Do the JAX interpolation.
    interp_obj = BicubicInterpolator(x_vals, y_vals, z_vals)
    res_jax = interp_obj(x_q, y_q)

    assert jnp.allclose(expected, res_jax, rtol=1e-3, atol=1e-4)


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
    input = jnp.asarray([-1.5, -1.1, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    expected = jnp.asarray([-0.0625, -0.0405, 1.0, 0.5625, 0.0, -0.0625, 0.0, 0.0])
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


def test_load_bicubic_interp_from_file(test_data_dir):
    """Test loading a bicubic interpolator object from a file."""
    filename = os.path.join(test_data_dir, "truncated-salt2-h17/fake_salt2_template_0.dat")
    test_obj = BicubicInterpolator.from_grid_file(filename)

    assert len(test_obj.x_vals) == 31
    assert len(test_obj.y_vals) == 701
    assert test_obj.z_vals.shape == (31, 701)
