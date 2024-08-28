"""Benchmarks for core TDAstro functionality.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

from tdastro.base_models import FunctionNode
from tdastro.util_nodes.np_random import NumpyRandomFunc


def time_chained_evaluate():
    """Time the generation of random numbers with an numpy generation node."""

    def _add_func(a, b):
        return a + b

    # Generate a starting mean and scale from uniform distributions. Use those to
    # generate a sample from the normal distribution. Then shift that sample by -5.0.
    loc_node = NumpyRandomFunc("uniform", low=10.0, high=20.0)
    scale_node = NumpyRandomFunc("uniform", low=0.5, high=1.0)
    norm_node = NumpyRandomFunc("normal", loc=loc_node, scale=scale_node)
    val_node = FunctionNode(_add_func, a=norm_node, b=-5.0)

    # Generate 100,000 samples.
    _ = val_node.sample_parameters(num_samples=100_000)
