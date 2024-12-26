"""Samplers used for testing that produces precomputed results. These
can be used in testing to produce known results or to use data previously
sampled from another method (such as pzflow).
"""

import numpy as np
import pandas as pd
from astropy.table import Table

from tdastro.base_models import FunctionNode


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


class TableSampler(FunctionNode):
    """A FunctionNode that returns values from a table, including
    a Pandas DataFrame or AstroPy Table.

    Attributes
    ----------
    data : pandas.DataFrame, astropy.table.Table, or dict
        The object containing the data to sample.
    columns : list of str
        The column names for the output columns.
    next_ind : int
        The next index to sample.
    """

    def __init__(self, data, node_label=None, **kwargs):
        self.next_ind = 0

        if isinstance(data, dict):
            self.data = pd.DataFrame(data)
        elif isinstance(data, Table):
            self.data = data.to_pandas()
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise TypeError("Unsupported data type for TableSampler.")

        # Add each of the flow's data columns as an output parameter.
        self.columns = [x for x in self.data.columns]
        super().__init__(self._non_func, node_label=node_label, outputs=self.columns, **kwargs)

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
        # Check that we have enough points left to sample.
        end_index = self.next_ind + graph_state.num_samples
        if end_index > len(self.data):
            raise IndexError()

        # Extract the table for [self.next_ind, end_index) and move
        # the index counter.
        samples = self.data[self.next_ind : end_index]
        self.next_ind = end_index

        # Parse out each column in the flow samples as a result vector.
        results = []
        for attr_name in self.columns:
            attr_values = samples[attr_name].values
            if graph_state.num_samples == 1:
                results.append(attr_values[0])
            else:
                results.append(np.array(attr_values))

        # Save and return the results.
        self._save_results(results, graph_state)
        return results
