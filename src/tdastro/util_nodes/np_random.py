"""Wrapper classes for calling numpy random number generators."""

from os import urandom

import numpy as np

from tdastro.base_models import FunctionNode


class NumpyRandomFunc(FunctionNode):
    """The base class for numpy random number generators.

    Attributes
    ----------
    func_name : `str`
        The name of the random function to use.
    _rng : `numpy.random._generator.Generator`
        This object's random number generator.

    Parameters
    ----------
    func_name : `str`
        The name of the random function to use.
    seed : `int`, optional
        The seed to use.

    Notes
    -----
    Since we need to create a new random number generator for this object
    and use that generator's functions, we cannot pass in the function directly.
    Instead we need to pass in the function's name.

    Examples
    --------
    # Create a uniform random number generator between 100.0 and 150.0
    func_node = NumpyRandomFunc("uniform", low=100.0, high=150.0)

    # Create a normal random number generator with mean=5.0 and std=1.0
    func_node = NumpyRandomFunc("normal", loc=5.0, scale=1.0)
    """

    def __init__(self, func_name, seed=None, **kwargs):
        self.func_name = func_name

        # Get a default random number generator for this object, using the
        # given seed if one is provided.
        if seed is None:
            seed = int.from_bytes(urandom(4), "big")
        self._rng = np.random.default_rng(seed=seed)

        # Check that the function exists in numpy's random number generator library.
        if not hasattr(self._rng, func_name):
            raise ValueError(f"Random function {func_name} does not exist.")
        func = getattr(self._rng, func_name)
        super().__init__(func, **kwargs)

    def compute(self, graph_state, given_args=None, rng_info=None, **kwargs):
        """Execute the wrapped function.

        The input arguments are taken from the current graph_state and the outputs
        are written to graph_state.

        Parameters
        ----------
        graph_state : `dict`
            A dictionary of dictionaries mapping node->hash, variable_name to value.
            This data structure is modified in place to represent the current state.
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        rng_info : `dict`, optional
            A dictionary of random number generator information for each node, such as
            the JAX keys or the numpy rngs.
        **kwargs : `dict`, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.

        Raises
        ------
        ``ValueError`` is ``func`` attribute is ``None``.
        """
        # If we are given a numpy random number generator, use that for this sample.
        if rng_info is not None and self.node_hash in rng_info:
            old_func = self.func
            self.func = getattr(rng_info[self.node_hash], self.func_name)
            result = super().compute(graph_state, given_args, rng_info, **kwargs)
            self.func = old_func
        else:
            result = super().compute(graph_state, given_args, rng_info, **kwargs)
        return result

    def generate(self, given_args=None, rng_info=None, **kwargs):
        """A helper function for testing that regenerates the output.

        Parameters
        ----------
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        rng_info : `dict`, optional
            A dictionary of random number generator information for each node, such as
            the JAX keys or the numpy rngs.
        **kwargs : `dict`, optional
            Additional function arguments.
        """
        state = self.sample_parameters(given_args, rng_info)
        return self.compute(state, given_args, rng_info, **kwargs)
