"""Wrapper classes for calling JAX random number generators."""

import jax.random

from tdastro.base_models import FunctionNode


def build_jax_keys(node_hashes, base_key):
    """Construct a dictionary of node's hash value to the JAX key.

    Parameters
    ----------
    node_hashes : iterable
        All of the node hash values as constructed by hashing the nodes' node_string.
    base_key : int or `jax._src.prng.PRNGKeyArray`
        The key on which to base the keys for the individual node's.

    Returns
    -------
    keys : `dict`
        A dictionary mapping each node's hash value to a unique JAX key.
    """
    keys = {}
    for value in node_hashes:
        new_seed = (value + base_key) % (2**32)
        keys[value] = jax.random.key(new_seed)
    return keys


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
        self._key = jax.random.key(seed)
        super().__init__(func, **kwargs)

    def compute(self, graph_state, given_args=None, rng_info=None, **kwargs):
        """Execute the wrapped JAX sampling function.

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

        Raises
        ------
        ``ValueError`` is ``func`` attribute is ``None``.
        """
        if self.func is None:
            raise ValueError(
                "func parameter is None for a JAXRandom. You need to either "
                "set func or override compute()."
            )

        # If a key is provided, use (and cycle) that. Otherwise use the internal seed≥
        if rng_info is not None and self.node_hash in rng_info:
            next_key, current_key = jax.random.split(rng_info[self.node_hash])
            rng_info[self.node_hash] = next_key
        else:
            next_key, current_key = jax.random.split(self._key)
            self._key = next_key

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

        # Generate the results.
        results = float(self.func(current_key, **args))
        graph_state[self.node_hash][self.outputs[0]] = results
        return results

    def generate(self, given_args=None, rng_info=None, **kwargs):
        """A helper function for testing that regenerates the parameters.

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

        self.jax_func = JaxRandomFunc(jax.random.normal, seed, **kwargs)
        super().__init__(_shift_and_scale, value=self.jax_func, loc=loc, scale=scale)

    def generate(self, given_args=None, rng_info=None, **kwargs):
        """A helper function for testing that regenerates the parameters.

        Parameters
        ----------
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        rng_info : `dict`, optional
            A dictionary of random number generator information for each node, such as
            the JAX keys or the numpy rngs.
        **kwargs : `dict`, optional
            Any additional keyword arguments.
        """
        state = self.sample_parameters(given_args, rng_info)
        return self.compute(state, given_args, rng_info, **kwargs)
