"""The BicubicInterpolator is used by SALT models.

It is adapted from sncosmo's BicubicInterpolator class (but implemented in JAX):
https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/salt2utils.pyx
"""

import jax.numpy as jnp

from jax import jit, vmap
from jax.lax import cond


@jit
def _kernel_value(input_vals):
    """A vectorized (JAX-ized) form of the kernval method from:
    https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/salt2utils.pyx
    
    Uses:
        A = -0.5
        B = A + 2.0 = 1.5
        C = A + 3.0 = 2.5
    """
    x = jnp.abs(input_vals)

    # Start with the case of 1.0 <= x <= 2.0.
    result = -0.5 * (-4.0 + x * (8.0 + x * (-5.0 + x)))

    # Override cases where x < 1.0
    result = jnp.where(input_vals < 1.0, x * x * (1.5 * x - 2.5) + 1.0, result)

    # Override cases where x > 2.0
    result = jnp.where(input_vals > 2.0, 0.0, result)
    return result


@jit
def find_indices(x_vals, x_q):
    """Finds the first index *after* each of the query values with
    the n - 1 index in each dimension mapped to n - 2.

    x_vals[ix[i]] <= x_q[i] < x_vals[ix[i] + 1] 
    for all i where ix[i] > 0 and ix[i] < n - 2.

    Parameters
    ----------
    x_vals : `jaxlib.xla_extension.ArrayImpl`
        The x coordinates of to search.
    x_q : `jaxlib.xla_extension.ArrayImpl`
        The x coordinates of the query points.
        
    Returns
    -------
    ix : `jaxlib.xla_extension.ArrayImpl`
        An array of the index values.
    """
    n_x = len(x_vals)
    ix = jnp.searchsorted(x_vals, x_q, side="right") - 1
    ix = jnp.where(ix < 0, 0, ix)
    ix = jnp.where(ix >= n_x - 1, n_x - 2, ix)
    return ix


class BicubicInterpolator:
    """An object that performs bicubic interpolation over a 2-d grid.
    
    Attributes
    ----------
    x_vals : `jaxlib.xla_extension.ArrayImpl`
        The values along the x-axis of the grid.
    y_vals : `jaxlib.xla_extension.ArrayImpl`
        The values along the y-axis of the grid.
    z_vals : `jaxlib.xla_extension.ArrayImpl`
        The values along the z-axis of the grid.
    """

    def __init__(self, x_vals, y_vals, z_vals):
        # Load and validate the x values.
        self.x_vals = jnp.asarray(x_vals)
        if len(self.x_vals.shape) != 1:
            raise ValueError(f"x values should be a 1-d array. Found shape={self.x_vals.shape}")
        if len(self.x_vals) == 0:
            raise ValueError("Empty x array given")
        if jnp.any(self.x_vals[:-1] >= self.x_vals[1:]):
            raise ValueError("x values must be in sorted order.")
        self.n_x = len(self.x_vals)
        self.x_step_size = jnp.append(self.x_vals[1:]-self.x_vals[:-1], jnp.asarray([1.0, 1.0]))

        # Load and validate the y values.
        self.y_vals = jnp.asarray(y_vals)
        if len(self.y_vals.shape) != 1:
            raise ValueError(f"y values should be a 1-d array. Found shape={self.y_vals.shape}")
        if len(self.y_vals) == 0:
            raise ValueError("Empty y array given")
        if jnp.any(self.y_vals[:-1] >= self.y_vals[1:]):
            raise ValueError("y values must be in sorted order.")      
        self.n_y = len(self.y_vals)
        self.y_step_size = jnp.append(self.y_vals[1:]-self.y_vals[:-1], jnp.asarray([1.0, 1.0]))

        self.z_vals = jnp.asarray(z_vals)
        if len(self.z_vals.shape) != 2:
            raise ValueError(f"z values should be a 2-d array. Found shape={self.x_vals.shape}")
        if self.z_vals.shape[0] != len(self.x_vals) or self.z_vals.shape[1] != len(self.y_vals):
            raise ValueError(
                f"z values wrong shape. Expected shape=({len(self.x_vals)}, {len(self.y_vals)})."
                f" Found shape={self.x_vals.shape}."
            )
        
        self.linear = jit(self._eval_linear)
        self.cubic = jit(self._eval_cubic)
        self.eval_row = vmap(jit(self.compute_row))

    def _eval_linear(self, x_q, y_q, ix, iy):
        ax = ((x_q - self.x_vals[ix]) / (self.x_vals[ix+1] - self.x_vals[ix]))
        ay = ((y_q - self.y_vals[iy]) / (self.y_vals[iy+1] - self.y_vals[iy]))
        ay2 = 1.0 - ay

        return (
            (1.0 - ax) * (ay2 * self.z_vals[ix][iy] + ay * self.z_vals[ix][iy+1]) +
            ax * (ay2 * self.z_vals[ix+1][iy] + ay * self.z_vals[ix+1][iy+1])
        )

    def _eval_cubic(self, x_q, y_q, ix, iy):
        dx = (self.x_vals[ix] - x_q) / (self.x_vals[ix+1] - self.x_vals[ix])
        dy = (self.y_vals[iy] - y_q) / (self.y_vals[iy+1] - self.y_vals[iy])
        wx = _kernel_value(jnp.asarray([dx-1.0, dx, dx+1.0, dx+2.0]))
        wy = _kernel_value(jnp.asarray([dy-1.0, dy, dy+1.0, dy+2.0]))
        return (
            wx[0] * (
                wy[0] * self.z_vals[ix-1][iy-1] +
                wy[1] * self.z_vals[ix-1][iy] +
                wy[2] * self.z_vals[ix-1][iy+1] +
                wy[3] * self.z_vals[ix-1][iy+2]
            ) +
            wx[1] * (
                wy[0] * self.z_vals[ix][iy-1] +
                wy[1] * self.z_vals[ix][iy] +
                wy[2] * self.z_vals[ix][iy+1] +
                wy[3] * self.z_vals[ix][iy+2]
            ) +
            wx[2] * (
                wy[0] * self.z_vals[ix+1][iy-1] +
                wy[1] * self.z_vals[ix+1][iy] +
                wy[2] * self.z_vals[ix+1][iy+1] +
                wy[3] * self.z_vals[ix+1][iy+2]
            ) +
            wx[3] * (
                wy[0] * self.z_vals[ix+2][iy-1] +
                wy[1] * self.z_vals[ix+2][iy] +
                wy[2] * self.z_vals[ix+2][iy+1] +
                wy[3] * self.z_vals[ix+2][iy+2]
            )
        )

    def compute_row(self, x_q, y_q, ix, iy, xflag, yflag):
        results = cond(
            (xflag == -1) | (yflag == -1),
            lambda x_q, y_q, ix, iy, xflag, yflag : 0.0,
            lambda x_q, y_q, ix, iy, xflag, yflag : cond(
                (xflag == 1) & (yflag == 1),
                self.cubic,
                self.linear,
                x_q,
                y_q,
                ix,
                iy,
            ),
            x_q,
            y_q,
            ix,
            iy,
            xflag,
            yflag,
        )
        return results

    def __call__(self, x_q, y_q):
        x_q = jnp.asarray(x_q)
        y_q = jnp.asarray(y_q)
        n_xq = len(x_q)
        n_yq = len(y_q)

        # Find the first index *after* each of the query values with
        # the n-1 index in each dimension mapped to n-2.
        ix = find_indices(self.x_vals, x_q)
        iy = find_indices(self.y_vals, y_q)

        # Create the flags for which rows of each input are outside the bounds (-1),
        # need linear interpolation (0), and can use cubic interpolation (1).
        # Compute flags for each query indicating out of bounds (-1), use linear interpolation (0)
        # or use cubic interpolation (1).
        xflag = jnp.where(jnp.full(n_xq, self.n_x < 3) | (ix == 0) | (ix > (self.n_x - 3)), 0, 1)
        xflag = jnp.where((x_q >= self.x_vals[0]) & (x_q <= self.x_vals[-1]), xflag, -1)
        yflag = jnp.where(jnp.full(n_yq, self.n_y < 3) | (iy == 0) | (iy > (self.n_y - 3)), 0, 1)
        yflag = jnp.where((y_q >= self.y_vals[0]) & (y_q <= self.y_vals[-1]), yflag, -1)

        # Create flattened matrix versions of the key information.
        xq_all = jnp.ravel(jnp.tile(jnp.asarray(x_q), (len(y_q), 1)).T)
        yq_all = jnp.ravel(jnp.tile(jnp.asarray(y_q), (len(x_q), 1)))
        ix_all = jnp.ravel(jnp.tile(ix, (len(y_q), 1)).T)
        iy_all = jnp.ravel(jnp.tile(iy, (len(x_q), 1)))
        xflag_all = jnp.ravel(jnp.tile(xflag, (n_yq, 1)).T)
        yflag_all = jnp.ravel(jnp.tile(yflag, (n_xq, 1)))

        # Use the vmapped function to evaluate every combination of x and y points.
        results = self.eval_row(xq_all, yq_all, ix_all, iy_all, xflag_all, yflag_all)
        results = results.reshape((n_xq, n_yq))
        return results
