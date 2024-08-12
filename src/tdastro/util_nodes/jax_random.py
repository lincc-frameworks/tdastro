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
        self._fixed_seed = seed
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
            self._key = jax.random.key(self._object_seed)

    def compute(self, graph_state, given_args=None, **kwargs):
        """Execute the wrapped JAX sampling function.

        Parameters
        ----------
        graph_state : `dict`
            A dictionary of dictionaries mapping node->hash, variable_name to value.
            This data structure is modified in place to represent the current state.
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
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

        # Build a dictionary of arguments for the function.
        args = {}
        for key in self.arg_names:
            # Override with the given arg or kwarg in that order.
            if given_args is not None and self.setters[key].full_name in given_args:
                args[key] = given_args[self.setters[key].full_name]
            elif key in kwargs:
                args[key] = kwargs[key]
            else:
                args[key] = graph_state[self.node_hash][key]
        self._key, subkey = jax.random.split(self._key)

        results = float(self.func(subkey, **args))
        graph_state[self.node_hash][self.outputs[0]] = results
        return results

    def generate(self, **kwargs):
        """A helper function for testing that regenerates the parameters."""
        state = self.sample_parameters()
        return self.compute(state, **kwargs)


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

    def generate(self, **kwargs):
        """A helper function for testing that regenerates the parameters."""
        state = self.sample_parameters()
        return self.compute(state, **kwargs)
