"""A collection of sampled parameters from a statistic distribution.  Parameters are defined"""

import numpy as np


class GraphState:
    """A class to hold the state(s) of the each variable for one or more samples of the random
    variables in the graph. Each entry is index by a combination of node name and variable name.

    Attributes
    ----------
    states : `dict`
        A dictionary of dictionaries mapping node->hash, variable_name to either a
        value or array of values.
    num_samples : `int`
        A count of the number of samples stored in the GraphState.
    """

    def __init__(self, num_samples=1):
        if num_samples < 1:
            raise ValueError(f"Invalid number of samples {num_samples}")
        self.num_samples = num_samples
        self.num_parameters = 0
        self.states = {}

    def __len__(self):
        return self.num_parameters

    def __getitem__(self, key):
        """Access the dictionary of parameter values for a node name."""
        return self.states[key]

    def get_node_state(self, node_name, sample_num=0):
        """Get a dictionary of all parameters local to the given node
        for a single sample state.

        Parameters
        ----------
        node_name : `str`
            The parent node whose variables to extract.
        sample_num : `int`
            The number of sample to extract.

        Returns
        -------
        values : `dict`
            A dictionary mapping the parameter name to its value.
        """
        if node_name not in self.states:
            raise KeyError(f"Node name {node_name} not found.")
        if sample_num < 0 or sample_num >= self.num_samples:
            raise ValueError(f"Invalid index {sample_num} in GraphState with {self.num_samples} entries.")
        if self.num_samples == 1:
            values = self.states[node_name]
        else:
            values = {}
            for var_name, val in self.states[node_name].items():
                values[var_name] = val[sample_num]
        return values

    def set(self, node_name, var_name, value):
        """Set a (new) parameter's value(s) in the GraphState from a given constant value
        or an array of length num_samples (to set all the values at once).

        Parameters
        ----------
        node_name : `str`
            The parent node holding this variable.
        var_name : `str`
            The parameter's name.
        value : any
            The new value of the parameter.
        """
        # Update the meta data.
        if node_name not in self.states:
            self.states[node_name] = {}
        if var_name not in self.states[node_name]:
            self.num_parameters += 1

        # Set the actual values.
        if self.num_samples == 1:
            # If this GraphStae holds only a single sample, set it from the given value.
            self.states[node_name][var_name] = value
        elif hasattr(value, "__len__") and hasattr(value, "copy"):
            # If we are given an array of samples, confirm it is the correct length and use it.
            if len(value) != self.num_samples:
                raise ValueError(f"Incompatible number of samples {self.num_samples} vs {len(value)}.")
            self.states[node_name][var_name] = value.copy()
        else:
            # If the GraphState holds N samples and we got a single value, make an array of it.
            self.states[node_name][var_name] = np.full((self.num_samples), value)

    def extract_single_sample(self, sample_num):
        """Create a new GraphState with a single sample state.

        Parameters
        ----------
        sample_num : `int`
            The number of sample to extract.
        """
        if self.num_samples <= 0:
            raise ValueError("Cannot sample an empty GraphState")
        if sample_num < 0 or sample_num >= self.num_samples:
            raise ValueError(f"Invalid index {sample_num} in GraphState with {self.num_samples} entries.")

        # Make a copy of the GraphState with exactly one sample.
        new_state = GraphState(1)
        new_state.num_parameters = self.num_parameters
        for node_name in self.states:
            new_state.states[node_name] = {}
            for var_name, value in self.states[node_name].items():
                if self.num_samples == 1:
                    new_state.states[node_name][var_name] = value
                else:
                    new_state.states[node_name][var_name] = value[sample_num]
        return new_state
