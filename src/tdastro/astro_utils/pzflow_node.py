"""A wrapper around pzflow sampling.

For the full pzflow package see:
https://github.com/jfcrenshaw/pzflow
"""

import numpy as np
from pzflow import Flow

from tdastro.base_models import FunctionNode


class PZFlowNode(FunctionNode):
    """A node that wraps sampling from pzflow.

    Attributes
    ----------
    flow : pzflow.flow.Flow or pzflow.flowEnsemble.FlowEnsemble
        The object from which to sample.
    columns : list of str
        The column names for the output columns.
    """

    def __init__(self, flow_obj, node_label=None, **kwargs):
        self.flow = flow_obj

        # Add each of the flow's data columns as an output parameter.
        self.columns = [x for x in flow_obj.data_columns]
        super().__init__(self._non_func, node_label=node_label, outputs=self.columns, **kwargs)

    def _non_func(self):
        """This function does nothing. Everything happens in the overloaded compute()."""
        pass

    @classmethod
    def from_file(cls, filename, node_label=None):
        """Create a PZFlowNode from a saved flow file.

        Parameters
        ----------
        filename : str or Path
            The location of the saved flow.
        node_label : `str`
            An optional human readable identifier (name) for the current node.
        """
        flow_to_use = Flow(file=filename)
        return PZFlowNode(flow_to_use, node_label=node_label)

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
        # If a random number generator is used, use that to compute the seed.
        seed = None if rng_info is None else int.from_bytes(rng_info.bytes(4), byteorder="big")
        samples = self.flow.sample(graph_state.num_samples, seed=seed)

        # Parse out each column in the flow samples as a result vector.
        results = []
        for attr_name in self.flow.data_columns:
            attr_values = samples[attr_name].values
            if graph_state.num_samples == 1:
                results.append(attr_values[0])
            else:
                results.append(np.array(attr_values))

        # Save and return the results.
        self._save_results(results, graph_state)
        return results
