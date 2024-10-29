"""A sampler used for testing that produces known (given) results."""

import numpy as np

from tdastro.base_models import FunctionNode


class GivenSampler(FunctionNode):
    """A FunctionNode that returns given results.

    Attributes
    ----------
    values : `float`, `list, or `numpy.ndarray`
        The values to return.
    next_ind : int
        The index of the next value.
    """

    def __init__(self, values, **kwargs):
        self.values = np.array(values)
        self.next_ind = 0
        super().__init__(self._non_func, **kwargs)

    def _non_func(self):
        """This function does nothing. Everything happens in the overloaded compute()."""
        pass

    def reset(self):
        """Reset the next index to use."""
        self.next_ind = 0

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

        Parameters
        ----------
        graph_state : `GraphState`
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            Unused in this function, but included to provide consistency with other
            compute functions.
        **kwargs : `dict`, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        if graph_state.num_samples == 1:
            if self.next_ind >= len(self.values):
                raise IndexError()

            results = self.values[self.next_ind]
            self.next_ind += 1
        else:
            end_ind = self.next_ind + graph_state.num_samples
            if end_ind > len(self.values):
                raise IndexError()

            results = self.values[self.next_ind : end_ind]
            self.next_ind = end_ind

        # Save and return the results.
        self._save_results(results, graph_state)
        return results
