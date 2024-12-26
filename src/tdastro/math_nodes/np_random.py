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

    def set_seed(self, new_seed):
        """Update the random number generator's seed to a given value.

        Parameters
        ----------
        new_seed : `int`
            The given seed
        """
        self._rng = np.random.default_rng(seed=new_seed)
        self.func = getattr(self._rng, self.func_name)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Execute the wrapped function.

        The input arguments are taken from the current graph_state and the outputs
        are written to graph_state.

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

        Raises
        ------
        ``ValueError`` is ``func`` attribute is ``None``.
        """
        args = self._build_inputs(graph_state, **kwargs)
        num_samples = None if graph_state.num_samples == 1 else graph_state.num_samples

        # If a random number generator is given use that. Otherwise use the default one.
        if rng_info is not None:
            func = getattr(rng_info, self.func_name)
            results = func(**args, size=num_samples)
        else:
            results = self.func(**args, size=num_samples)
        self._save_results(results, graph_state)
        return results
