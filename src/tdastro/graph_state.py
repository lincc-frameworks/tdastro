"""A collection of sampled parameters from a statistic distribution.

Model parameters are random variables that are sampled together in a joint distribution
using a graph of dependencies. For example the functions:

f(a, b) = x
g(c) = y
h(x, y) = z

indicate that x depends on a and b, y depends on c, and z depends on x and y (and thus on a, b, and c
as well). These would form a graph that looks like:

a -\
    x - \
b -/     \
          z
c -- y -- /

Within TDAstro, variables are grouped into logical sets called nodes. The combination of node name
and variable name are used to indicate specific values, allowing us to use the same variable names
in multiple nodes. For example a node to generate samples from a Gaussian distribution may have internal
parameters called mean and scale that might take on different values depending on what the node is generating.
We could have one mean for an object's brightness and another for it's positional relative to the center
of a host galaxy.
"""

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

    def set(self, node_name, var_name, value, force_copy=False):
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
        force_copy : `bool`
            Make a copy of data in an array. If set to ``False`` this will link
            to the array, saving memory and computation time.
            Default: ``False``
        """
        # Update the meta data.
        if node_name not in self.states:
            self.states[node_name] = {}
        if var_name not in self.states[node_name]:
            self.num_parameters += 1

        # Set the actual values.
        if self.num_samples == 1:
            # If this GraphState holds only a single sample, set it from the given value.
            self.states[node_name][var_name] = value
        elif hasattr(value, "__len__") and hasattr(value, "copy"):
            # If we are given an array of samples, confirm it is the correct length and use it.
            if len(value) != self.num_samples:
                raise ValueError(f"Incompatible number of samples {self.num_samples} vs {len(value)}.")
            if force_copy:
                self.states[node_name][var_name] = value.copy()
            else:
                self.states[node_name][var_name] = value
        else:
            # If the GraphState holds N samples and we got a single value, make an array of it.
            self.states[node_name][var_name] = np.full((self.num_samples), value)

    def update(self, inputs, force_copy=False):
        """Set multiple parameters' value in the GraphState from a GraphState or a
        dictionary of the same form.

        Parameters
        ----------
        inputs : `GraphState` or `dict`
            Values to copy.
        force_copy : `bool`
            Make a copy of data in an array. If set to ``False`` this will link
            to the array, saving memory and computation time.
            Default: ``False``
        """
        if isinstance(inputs, GraphState):
            if self.num_samples != inputs.num_samples:
                raise ValueError("GraphSates must have the same number of samples.")
            new_states = inputs.states
        else:
            new_states = inputs

        # Set the values one by one.
        for node_name, node_vars in new_states.items():
            for var_name, value in node_vars.items():
                self.set(node_name, var_name, value, force_copy=force_copy)

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


def transpose_dict_of_list(input_dict, num_elem):
    """Transpose a dictionary of iterables to a list of dictionaries.

    Parameters
    ----------
    input_dict : `dict`
        A dictionary of iterables, each of which is length num_elem.
    num_elem : `int`
        The length of the iterables.

    Returns
    -------
    output_list : `list`
        A length num_elem list of dictionaries, each with the same keys mapping
        to a single value.

    Raises
    ------
    ``ValueError`` if any of the iterables have different lengths.
    """
    if num_elem < 1:
        raise ValueError(f"Trying to transpose a dictionary with {num_elem} elements")

    output_list = [{} for _ in range(num_elem)]
    for key, values in input_dict.items():
        if len(values) != num_elem:
            raise ValueError(f"Entry {key} has length {len(values)}. Expected {num_elem}.")
        for i in range(num_elem):
            output_list[i][key] = values[i]
    return output_list