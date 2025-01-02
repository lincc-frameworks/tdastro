"""Samplers used for testing that produces precomputed results. These
can be used in testing to produce known results or to use data previously
sampled from another method (such as pzflow).
"""

import numpy as np
import pandas as pd
from astropy.table import Table

from tdastro.base_models import FunctionNode
from tdastro.math_nodes.np_random import NumpyRandomFunc


class GivenValueList(FunctionNode):
    """A FunctionNode that returns given results for a single parameter.

    Attributes
    ----------
    values : `float`, `list, or `numpy.ndarray`
        The values to return.
    next_ind : int
        The index of the next value.
    """

    def __init__(self, values, **kwargs):
        self.values = np.asarray(values)
        if len(values) == 0:
            raise ValueError("No values provided for GivenValueList")
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


class GivenValueSampler(NumpyRandomFunc):
    """A FunctionNode that returns randomly selected items from a given list
    with replacement.

    Attributes
    ----------
    values : list or numpy.ndarray
        The values to select from.
    _num_values : int
        The number of values that can be sampled.
    """

    def __init__(self, values, seed=None, **kwargs):
        self.values = np.asarray(values)
        self._num_values = len(values)
        if self._num_values == 0:
            raise ValueError("No values provided for NumpySamplerNode")

        super().__init__("uniform", seed=seed, **kwargs)

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

        Parameters
        ----------
        graph_state : `GraphState`
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : `dict`, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        rng = rng_info if rng_info is not None else self._rng

        if graph_state.num_samples == 1:
            inds = rng.integers(0, self._num_values)
        else:
            inds = rng.integers(0, self._num_values, size=graph_state.num_samples)

        return self.values[inds]


class TableSampler(NumpyRandomFunc):
    """A FunctionNode that returns values from a table-like data,
    including a Pandas DataFrame or AstroPy Table. The results returned
    can be in-order (for testing) or randomly selected with replacement.

    Parameters
    ----------
    data : pandas.DataFrame, astropy.table.Table, or dict
        The object containing the data to sample.
    in_order : bool
        Return the given data in order of the rows (True). If False, performs
        random sampling with replacement. Default: False

    Attributes
    ----------
    data : astropy.table.Table
        The object containing the data to sample.
    in_order : bool
        Return the given data in order of the rows (True). If False, performs
        random sampling with replacement. Default: False
    next_ind : int
        The next index to sample for in order sampling.
    num_values : int
        The total number of items from which to draw the data.
    """

    def __init__(self, data, in_order=False, **kwargs):
        self.next_ind = 0
        self.in_order = in_order

        if isinstance(data, dict):
            self.data = Table(data)
        elif isinstance(data, Table):
            self.data = data.copy()
        elif isinstance(data, pd.DataFrame):
            self.data = Table.from_pandas(data)
        else:
            raise TypeError("Unsupported data type for TableSampler.")

        # Check there are some rows.
        self._num_values = len(self.data)
        if self._num_values == 0:
            raise ValueError("No data provided to TableSampler.")

        # Add each of the flow's data columns as an output parameter.
        super().__init__("uniform", outputs=self.data.colnames, **kwargs)

    def reset(self):
        """Reset the next index to use. Only used for in-order sampling."""
        self.next_ind = 0

    def compute(self, graph_state, rng_info=None, **kwargs):
        """Return the given values.

        Parameters
        ----------
        graph_state : `GraphState`
            An object mapping graph parameters to their values. This object is modified
            in place as it is sampled.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : `dict`, optional
            Additional function arguments.

        Returns
        -------
        results : any
            The result of the computation. This return value is provided so that testing
            functions can easily access the results.
        """
        # Compute the indices to sample.
        if self.in_order:
            # Check that we have enough points left to sample.
            end_index = self.next_ind + graph_state.num_samples
            if end_index > len(self.data):
                raise IndexError()

            sample_inds = np.arange(self.next_ind, end_index)
            self.next_ind = end_index
        else:
            rng = rng_info if rng_info is not None else self._rng
            sample_inds = rng.integers(0, self._num_values, size=graph_state.num_samples)

        # Parse out each column into a separate parameter with the column name as its name.
        results = []
        for attr_name in self.outputs:
            attr_values = np.asarray(self.data[attr_name][sample_inds])
            if graph_state.num_samples == 1:
                results.append(attr_values[0])
            else:
                results.append(attr_values)

        # Save and return the results.
        self._save_results(results, graph_state)
        return results
