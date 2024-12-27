"""Samplers used for generating (RA, dec) coordinates."""

import numpy as np

from tdastro.math_nodes.given_sampler import TableSampler
from tdastro.math_nodes.np_random import NumpyRandomFunc


class UniformRADEC(NumpyRandomFunc):
    """A FunctionNode that uniformly samples (RA, dec) over a sphere,

    Attributes
    ----------
    use_degrees : bool
        The default return unit. If True returns samples in degrees.
        Otherwise, if False, returns samples in radians.
    """

    def __init__(self, outputs=None, seed=None, use_degrees=True, **kwargs):
        self.use_degrees = use_degrees

        # Override key arguments. We create a uniform sampler function, but
        # won't need it because the subclass overloads compute().
        func_name = "uniform"
        outputs = ["ra", "dec"]
        super().__init__(func_name, outputs=outputs, seed=seed, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

        Parameters
        ----------
        graph_state : `GraphState`
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : `dict`, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        rng = rng_info if rng_info is not None else self._rng

        # Generate the random (RA, dec) lists.
        ra = rng.uniform(0.0, 2.0 * np.pi, size=graph_state.num_samples)
        dec = np.arccos(2.0 * rng.uniform(0.0, 1.0, size=graph_state.num_samples) - 1.0) - (np.pi / 2.0)
        if self.use_degrees:
            ra = np.degrees(ra)
            dec = np.degrees(dec)

        # If we are generating a single sample, return floats.
        if graph_state.num_samples == 1:
            ra = ra[0]
            dec = dec[0]

        # Set the outputs and return the results. This takes the place of
        # function node's _save_results() function because we know the outputs.
        graph_state.set(self.node_string, "ra", ra)
        graph_state.set(self.node_string, "dec", dec)
        return [ra, dec]


class OpSimRADECSampler(TableSampler):
    """A FunctionNode that samples RA and dec (and time) from an OpSim.
    RA and dec are returned in degrees.

    Parameters
    ----------
    data : OpSim
        The OpSim object to use for sampling.
    in_order : bool
        Return the given data in order of the rows (True). If False, performs
        random sampling with replacement. Default: False
    """

    def __init__(self, data, in_order=False, **kwargs):
        data_dict = {
            "ra": data["ra"],
            "dec": data["dec"],
            "time": data["time"],
        }
        super().__init__(data_dict, in_order=in_order, **kwargs)
