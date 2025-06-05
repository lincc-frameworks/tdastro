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
        rng = rng_info if rng_info is not None else self._rng

        # Generate the random (RA, dec) lists.
        ra = rng.uniform(0.0, 2.0 * np.pi, size=graph_state.num_samples)
        dec = np.arcsin(rng.uniform(-1.0, 1.0, size=graph_state.num_samples))
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

    Note
    ----
    Does not currently use uniform sampling from the radius. Uses a very
    rough approximate as a proof of concept. Do not use for statistical analysis.

    Parameters
    ----------
    data : OpSim
        The OpSim object to use for sampling.
    radius : float
        The radius of the observations in degrees. Use 0.0 to just sample
        the centers of the images. Default: 0.0
    in_order : bool
        Return the given data in order of the rows (True). If False, performs
        random sampling with replacement. Default: False
    """

    def __init__(self, data, radius=0.0, in_order=False, **kwargs):
        if radius < 0.0:
            raise ValueError("Invalid radius: {radius}")
        self.radius = radius

        data_dict = {
            "ra": data["ra"],
            "dec": data["dec"],
            "time": data["time"],
        }
        super().__init__(data_dict, in_order=in_order, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

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
        # Sample the center RA, dec, and times without the radius.
        results = super().compute(graph_state, rng_info=rng_info, **kwargs)

        if self.radius > 0.0:
            # Add an offset around the center. This is currently a placeholder that does
            # NOT produce a uniform sampling. TODO: Make this uniform sampling.
            rng = rng_info if rng_info is not None else self._rng

            # Choose a uniform circle around the center point. Not that this is not uniform over
            # the final RA, dec because it does not account for compression in dec around the polls.
            offset_amt = self.radius * np.sqrt(rng.uniform(0.0, 1.0, size=graph_state.num_samples))
            offset_ang = 2.0 * np.pi * rng.uniform(0.0, 1.0, size=graph_state.num_samples)

            # Add the offsets to RA and dec. Keep time unchanged.
            results[0] += offset_amt * np.cos(offset_ang)  # RA
            results[1] += offset_amt * np.sin(offset_ang)  # dec

            # Resave the results (overwriting the previous results)
            self._save_results(results, graph_state)

        return results


class OpSimUniformRADECSampler(NumpyRandomFunc):
    """A FunctionNode that samples RA and dec uniformly from the area covered
    by an OpSim.  RA and dec are returned in degrees.

    Note
    ----
    This uses rejection sampling and can be quite slow for small coverage.

    Attributes
    ----------
    data : OpSim
        The OpSim object to use for sampling.
    radius : float
        The radius of the observations in degrees. Must be > 0.0.
        Default: 1.0
    max_iteraions : int
        The maximum number of iterations to perform. Default: 1000
    """

    def __init__(self, data, radius=1.0, outputs=None, seed=None, max_iteraions=1000, **kwargs):
        if radius <= 0.0:
            raise ValueError("Invalid radius: {radius}")
        self.radius = radius

        if len(data) == 0:
            raise ValueError("OpSim data cannot be empty.")
        self.data = data

        if max_iteraions <= 0:
            raise ValueError("Invalid max_iteraions: {max_iteraions}")
        self.max_iteraions = max_iteraions

        # Override key arguments. We create a uniform sampler function, but
        # won't need it because the subclass overloads compute().
        func_name = "uniform"
        outputs = ["ra", "dec"]
        super().__init__(func_name, outputs=outputs, seed=seed, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

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
        (ra, dec) : tuple of floats or np.ndarray
            If a single sample is generated, returns a tuple of floats. Otherwise,
            returns a tuple of np.ndarrays.
        """
        rng = rng_info if rng_info is not None else self._rng

        ra = np.zeros(graph_state.num_samples)
        dec = np.zeros(graph_state.num_samples)
        mask = np.full(graph_state.num_samples, False)
        num_missing = graph_state.num_samples

        # Rejection sampling to ensure the samples are within the OpSim coverage.
        # This can take many iterations if the coverage is small.
        iter_num = 1
        while num_missing > 0 and iter_num < 1000:
            # Generate new samples for the missing ones.
            ra[~mask] = np.degrees(rng.uniform(0.0, 2.0 * np.pi, size=num_missing))
            dec[~mask] = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=num_missing)))

            # Check if the samples are within the OpSim coverage.
            mask = np.asarray(self.data.is_observed(ra, dec, self.radius))
            num_missing = np.sum(~mask)
            iter_num += 1

        # If we are generating a single sample, return floats.
        if graph_state.num_samples == 1:
            ra = ra[0]
            dec = dec[0]

        # Set the outputs and return the results. This takes the place of
        # function node's _save_results() function because we know the outputs.
        graph_state.set(self.node_string, "ra", ra)
        graph_state.set(self.node_string, "dec", dec)
        return (ra, dec)
