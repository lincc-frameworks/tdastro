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

    def __init__(self, func_name, **kwargs):
        self.func_name = func_name
        self._rng = np.random.default_rng()
        if not hasattr(self._rng, func_name):
            raise ValueError(f"Random function {func_name} does not exist.")
        func = getattr(self._rng, func_name)
        super().__init__(func, **kwargs)

    def set_graph_base_seed(self, graph_base_seed):
        """Set a new graph base seed.

        Notes
        -----
        WARNING: This seed should almost never be set manually. Using the same
        seed for multiple graph instances will produce biased samples.

        Parameters
        ----------
        graph_base_seed : `int`, optional
            A base random seed to use for this specific evaluation graph.
        """
        super().set_graph_base_seed(graph_base_seed)

        # We create a new random number generator with the new object seed and
        # link to that object's function.
        self._rng = np.random.default_rng(self._object_seed)
        self.func = getattr(self._rng, self.func_name)

    def compute(self, **kwargs):
        """Execute the wrapped JAX sampling function.

        Parameters
        ----------
        **kwargs : `dict`, optional
            Additional function arguments.
        """
        args = self._build_args_dict(**kwargs)
        return self.func(**args)
