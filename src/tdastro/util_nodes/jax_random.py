"""Wrapper classes for calling JAX random number generators."""

import jax.random

from tdastro.base_models import FunctionNode


class JaxRandomFunc(FunctionNode):
    """The base class for JAX random number generators.

    Attributes
    ----------
    _key : `jax._src.prng.PRNGKeyArray`

    Parameters
    ----------
    func : function
        The JAX function to sample.
    seed : `int`, optional
        The seed to use.

    Note
    ----
    Automatically splits keys each time ``compute()`` is called, so
    each call produces a new pseudorandom number.
    """

    def __init__(self, func, seed=None, **kwargs):
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
        old_seed = self._object_seed
        super().set_seed(new_seed, graph_base_seed)
        if old_seed != self._object_seed or force_update:
            self._key = jax.random.key(self._object_seed)

    def compute(self, **kwargs):
        """Execute the wrapped JAX sampling function.

        Parameters
        ----------
        **kwargs : `dict`, optional
            Additional function arguments.

        Raises
        ------
        ``ValueError`` is ``func`` attribute is ``None``.
        """
        if self.func is None:
            raise ValueError(
                "func parameter is None for a JAXRandom. You need to either "
                "set func or override compute()."
            )

        args = self._build_args_dict(**kwargs)
        self._key, subkey = jax.random.split(self._key)
        results = float(self.func(subkey, **args))
        self._save_result_parameters(results)
        return results

    def resample_and_compute(self, **kwargs):
        """Compute the value, forcing a resampling of the random variables."""
        self.sample_parameters(**kwargs)
        return self.compute(**kwargs)


class JaxRandomNormal(FunctionNode):
    """A wrapper for the JAX normal function that takes
    a mean and std.

    Parameters
    ----------
    loc : `float`
        The mean of the distribution.
    scale : `float`
        The std of the distribution.
    seed : `int`, optional
        The seed to use.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, loc, scale, seed=None, **kwargs):
        def _shift_and_scale(value, loc, scale):
            return scale * value + loc

        self.jax_func = JaxRandomFunc(jax.random.normal, **kwargs)

        super().__init__(_shift_and_scale, value=self.jax_func, loc=loc, scale=scale)

        # Set the graph base seed so that it propagates up to the JAX function.
        # TODO: Come up with a better way of setting seeds for components.
        if seed is not None:
            self.update_graph_information(new_graph_base_seed=seed)

    def resample_and_compute(self, **kwargs):
        """Compute the value, forcing a resampling of the random variables."""
        self.sample_parameters(**kwargs)
        return self.compute(**kwargs)
