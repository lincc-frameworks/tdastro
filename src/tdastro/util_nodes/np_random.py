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
