"""Wrapper classes for calling numpy random number generators."""

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
        self._fixed_seed = seed

        # Use a temporary random number generator to seed the function.
        self._rng = np.random.default_rng()
        if not hasattr(self._rng, func_name):
            raise ValueError(f"Random function {func_name} does not exist.")
        func = getattr(self._rng, func_name)
        super().__init__(func, **kwargs)

        # Overwrite the func attribute using the new seed.
        self.set_seed(new_seed=seed)

    def set_seed(self, new_seed=None, graph_base_seed=None, force_update=False):
        """Update the object seed to the new value based.

        The new value can be: 1) a given seed (new_seed), 2) a value computed from
        the graph's base seed (graph_base_seed) and the object's string representation,
        or a completely random seed (if neither option is set).

        WARNING: This seed should almost never be set manually. Using the duplicate
        seeds for multiple graph instances or runs will produce biased samples.

        Parameters
        ----------
        new_seed : `int`, optional
            The given seed
        graph_base_seed : `int`, optional
            A base random seed to use for this specific evaluation graph.
        force_update : `bool`
            Reset the random number generator even if the seed has not change.
            This should only be set to ``True`` for testing.
        """
        # If we have set a fixed seed for this node, use that.
        if new_seed is None and self._fixed_seed is not None:
            new_seed = self._fixed_seed

        old_seed = self._object_seed
        super().set_seed(new_seed, graph_base_seed)
        if old_seed != self._object_seed or force_update:
            self._rng = np.random.default_rng(seed=self._object_seed)
            self.func = getattr(self._rng, self.func_name)

    def generate(self, **kwargs):
        """A helper function for testing that regenerates the output."""
        state = self.sample_parameters()
        return self.compute(state, **kwargs)
