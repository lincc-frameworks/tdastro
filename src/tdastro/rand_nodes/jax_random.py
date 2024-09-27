"""Wrapper classes for calling JAX random number generators.

JAX random number generators are stateless and require a key to be passed in.
The JaxRandomFunc class uses the key to generate the sample and also rotates
the key so it can be used in future queries.
"""

from os import urandom

import jax.random

from tdastro.base_models import FunctionNode, ParameterizedNode


def build_jax_keys_from_hashes(node_hashes, base_seed=None):
    """Construct a dictionary mapping each node's hash value to a unique JAX key.

    Parameters
    ----------
    node_hashes : iterable
        All of the node hash values as constructed by hashing the nodes' node_string.
    base_seed : `int`
        The key on which to base the keys for the individual nodes.

    Returns
    -------
    keys : `dict`
        A dictionary mapping each node's hash value to a unique JAX key.

    Raises
    ------
    ``ValueError`` if duplicates appear in ``node_hashes``.
    """
    if base_seed is None:
        base_seed = base_seed = int.from_bytes(urandom(4), "big")

    # Create a key entry for each node.
    keys = {}
    for value in node_hashes:
        if value in keys:
            raise ValueError(f"Key collision for value {value}")

        new_seed = (value + base_seed) % (2**32)
        keys[value] = jax.random.key(new_seed)
    return keys


def build_jax_keys_from_nodes(nodes, base_seed=None):
    """Construct a dictionary a list of each node's hash value to a JAX key
    from the list of nodes.

    Parameters
    ----------
    nodes : iterable or `ParameterizedNode`
        All of the nodes.
    base_seed : `int`
        The key on which to base the keys for the individual nodes.

    Returns
    -------
    keys : `dict`
        A dictionary mapping each node's hash value to a unique JAX key.
    """
    if isinstance(nodes, ParameterizedNode):
        nodes = [nodes]

    # Recursively generate the list of nodes' hash values for all dependencies.
    seen_nodes = set()
    hash_list = []
    for node in nodes:
        hash_list.extend(node.get_all_node_info("node_hash", seen_nodes))
    return build_jax_keys_from_hashes(hash_list, base_seed)


class JaxRandomFunc(FunctionNode):
    """The base class for JAX random number generators.

    Parameters
    ----------
    func : function
        The JAX function to sample.

    Note
    ----
    Automatically splits keys each time ``compute()`` is called, so
    each call produces a new pseudorandom number.
    """

    def __init__(self, func, **kwargs):
        super().__init__(func, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Execute the wrapped JAX sampling function.

        Parameters
        ----------
        graph_state : `GraphState`
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
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
        if rng_info is None:
            raise ValueError("JAX random number generation requires a JAX key. None given.")
        if self.node_hash not in rng_info:
            raise ValueError(f"No JAX found for node hash {self.node_hash} in {rng_info}.")

        # Split the key and save the other have to use later.
        next_key, current_key = jax.random.split(rng_info[self.node_hash])
        rng_info[self.node_hash] = next_key

        # Generate the results.
        args = self._build_inputs(graph_state, **kwargs)
        if graph_state.num_samples == 1:
            results = float(self.func(current_key, **args))
        else:
            use_shape = [graph_state.num_samples]
            results = self.func(current_key, shape=use_shape, **args)
        graph_state.set(self.node_string, self.outputs[0], results)
        return results

    def generate(self, given_args=None, num_samples=1, rng_info=None, **kwargs):
        """A helper function for testing that regenerates the parameters.

        Parameters
        ----------
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        num_samples : `int`
            A count of the number of samples to compute.
            Default: 1
        rng_info : `dict`, optional
            A dictionary of random number generator information for each node, such as
            the JAX keys or the numpy rngs.
        **kwargs : `dict`, optional
            Additional function arguments.
        """
        state = self.sample_parameters(given_args, num_samples, rng_info)
        return self.compute(state, rng_info, **kwargs)


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

    def __init__(self, loc, scale, **kwargs):
        def _shift_and_scale(value, loc, scale):
            return scale * value + loc

        self.jax_func = JaxRandomFunc(jax.random.normal, **kwargs)
        super().__init__(_shift_and_scale, value=self.jax_func, loc=loc, scale=scale)

    def generate(self, given_args=None, num_samples=1, rng_info=None, **kwargs):
        """A helper function for testing that regenerates the parameters.

        Parameters
        ----------
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        num_samples : `int`
            A count of the number of samples to compute.
            Default: 1
        rng_info : `dict`, optional
            A dictionary of random number generator information for each node, such as
            the JAX keys or the numpy rngs.
        **kwargs : `dict`, optional
            Any additional keyword arguments.
        """
        state = self.sample_parameters(given_args, num_samples, rng_info)
        return self.compute(state, rng_info, **kwargs)
