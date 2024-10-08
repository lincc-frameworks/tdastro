"""Wrapper classes for calling numpy random number generators."""

from os import urandom

import numpy as np

from tdastro.base_models import FunctionNode, ParameterizedNode


def build_rngs_from_hashes(node_hashes, base_seed=None):
    """Construct a dictionary a list of each node's hash value to a
    numpy random number generator.

    Parameters
    ----------
    node_hashes : iterable
        All of the node hash values as constructed by hashing the nodes' node_string.
    base_seed : int
        The key on which to base the keys for the individual nodes.

    Returns
    -------
    rngs : `dict`
        A dictionary mapping each node's hash value to a unique numpy rng.

    Raises
    ------
    ``ValueError`` if duplicates appear in ``node_hashes``.
    """
    if base_seed is None:
        base_seed = base_seed = int.from_bytes(urandom(4), "big")

    # Create a key entry for each node.
    rngs = {}
    for value in node_hashes:
        if value in rngs:
            raise ValueError(f"Key collision for value {value}")

        new_seed = (value + base_seed) % (2**32)
        rngs[value] = np.random.default_rng(seed=new_seed)
    return rngs


def build_rngs_from_nodes(nodes, base_seed=None):
    """Construct a dictionary mapping each node's hash value to a
    numpy random number generator.

    Parameters
    ----------
    nodes : iterable or `ParameterizedNode`
        All of the nodes.
    base_seed : `int`
        The key on which to base the keys for the individual nodes.

    Returns
    -------
    keys : `dict`
        A dictionary mapping each node's hash value to a numpy random number generator.
    """
    if isinstance(nodes, ParameterizedNode):
        nodes = [nodes]

    # Recursively generate the list of nodes' hash values for all dependencies.
    seen_nodes = set()
    hash_list = []
    for node in nodes:
        hash_list.extend(node.get_all_node_info("node_hash", seen_nodes))
    return build_rngs_from_hashes(hash_list, base_seed)


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
        args = self._build_inputs(graph_state, **kwargs)
        num_samples = None if graph_state.num_samples == 1 else graph_state.num_samples

        # If a random number generator is given use that. Otherwise use the default one.
        if rng_info is not None:
            if self.node_hash not in rng_info:
                raise KeyError("Node's hash not found in rng_info")
            func = getattr(rng_info[self.node_hash], self.func_name)
            results = func(**args, size=num_samples)
        else:
            results = self.func(**args, size=num_samples)
        self._save_results(results, graph_state)
        return results

    def generate(self, given_args=None, num_samples=1, rng_info=None, **kwargs):
        """A helper function for testing that regenerates the output.

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
