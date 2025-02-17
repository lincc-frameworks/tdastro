"""A class that can be used to quickly sample from a one-dimensional distribution provided
as an empirical PDF (x points and their corrersponding PDF(x) values) and then sampled using
an interpolation of the inverse CDF function.
"""

from os import urandom

import numpy as np
from scipy.interpolate import interp1d

from tdastro.base_models import FunctionNode


class DistributionSampler(FunctionNode):
    """A node for sampling from an arbitrary one-dimensional distribution
    using inverse transformation sampling.

    Attributes
    ----------
    _rng : numpy.random._generator.Generator
        This object's random number generator.
    _inv_cdf : scipy.interpolate.interp1d
        The interpolation function for the inverse CDF. Maps values [0, 1]
        to the corresponding x values.

    Parameters
    ----------
    x : numpy.ndarray[float]
        The random variable (input to the density function)
    y : numpy.ndarray[float]
        The probability density function (output).
    kind : str
        The mode of interpolation for CDF function.
        Must be one of "linear", "nearest", "nearest-up", "zero", "slinear",
        "quadratic", "cubic", "previous", or "next".
        Default: "linear"
    """

    def __init__(self, x, y, kind="linear", seed=None, **kwargs):
        super().__init__(self._non_func, **kwargs)

        # Get a default random number generator for this object, using the
        # given seed if one is provided.
        if seed is None:
            seed = int.from_bytes(urandom(4), "big")
        self._rng = np.random.default_rng(seed=seed)

        # Validate the inputs.
        if len(x) != len(y):
            raise ValueError(f"Length of x ({len(x)}) and y ({len(y)}) must be equal.")
        if np.any(y < 0):
            raise ValueError("Probability density function (y) must be non-negative.")

        # Create the CDF values (and normalize it to 1) and then create an interpolation
        # of the inverse CDF function.
        cdf_samples = np.cumsum(y)
        cdf_samples /= cdf_samples[-1]
        self._inv_cdf = interp1d(cdf_samples, x, kind=kind, fill_value=0.0)

    def set_seed(self, new_seed):
        """Update the random number generator's seed to a given value.

        Parameters
        ----------
        new_seed : int
            The given seed
        """
        self._rng = np.random.default_rng(seed=new_seed)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Draw samples from the inverse PDF.

        The input arguments are taken from the current graph_state and the outputs
        are written to graph_state.

        Parameters
        ----------
        graph_state : GraphState
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        num_samples = None if graph_state.num_samples == 1 else graph_state.num_samples

        # Draw samples of the CDF [0, 1] and then use the inverse CDF spline to map those
        # back to the corresponding x values.
        cdf_points = self._rng.random(num_samples)
        x_samples = self._inv_cdf(cdf_points)
        self._save_results(x_samples, graph_state)
        return x_samples
