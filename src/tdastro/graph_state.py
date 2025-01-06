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
from astropy.io import ascii
from astropy.table import Table


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
    num_parameters : `int`
        The total number of parameters stored in a single sample within GraphState.
    fixed_vars : `dict`
        A dictionary mapping the node name to a set of the variable names that
        are fixed in this GraphState instance.
    """

    def __init__(self, num_samples=1):
        if num_samples < 1:
            raise ValueError(f"Invalid number of samples {num_samples}")
        self.num_samples = num_samples
        self.num_parameters = 0
        self.states = {}
        self.fixed_vars = {}

    def __len__(self):
        return self.num_parameters

    def __next__(self):
        return next(self._iterate())

    def __iter__(self):
        return self._iterate()

    def _iterate(self):
        """Returns a single sliced state."""
        for idx in range(self.num_samples):
            yield self.extract_single_sample(idx)

    def __contains__(self, key):
        if key in self.states:
            return True
        elif "." in key:
            tokens = key.split(".")
            if len(tokens) != 2:
                raise KeyError(f"Invalid GraphState key: {key}")
            return tokens[0] in self.states and tokens[1] in self.states[tokens[0]]
        elif len(self.states) == 1:
            # Special case when we have only a single node stored in the graph state.
            node_state = list(self.states.values())[0]
            if key in node_state:
                return True
        else:
            return False

    def __str__(self):
        str_lines = []
        for node_name, node_vars in self.states.items():
            str_lines.append(f"{node_name}:")
            for var_name, value in node_vars.items():
                str_lines.append(f"    {var_name}: {value}")
        return "\n".join(str_lines)

    def __eq__(self, other):
        if self.num_samples != other.num_samples:
            return False
        if len(self.states) != len(other.states):
            return False

        for node_name, node_params in self.states.items():
            # Check that this node exists in both GraphStates and has the same number
            # of parameters.
            if node_name not in other.states:
                return False
            other_params = other.states[node_name]
            if len(node_params) != len(other_params):
                return False

            # Check that the values of each parameter matches.
            for var_name, var_value in node_params.items():
                if var_name not in other_params:
                    return False
                if not np.allclose(var_value, other_params[var_name]):
                    return False

        # Check that the 'fixed' dictionary is the same.
        if self.fixed_vars != other.fixed_vars:
            return False

        return True

    def __getitem__(self, key):
        """Access the dictionary of parameter values for a node name. Allows
        access by both the pair of keys and the extended name."""
        if key in self.states:
            return self.states[key]
        elif "." in key:
            tokens = key.split(".")
            if len(tokens) != 2:
                raise KeyError(f"Invalid GraphState key: {key}")
            return self.states[tokens[0]][tokens[1]]
        elif len(self.states) == 1:
            # Special case when we have only a single node stored
            # in the graph state.
            node_state = list(self.states.values())[0]
            if key in node_state:
                return node_state[key]
        else:
            raise KeyError(f"Unknown GraphState key: {key}")

    @staticmethod
    def extended_param_name(node_name, param_name):
        """A helper function to create the full parameter name.

        Parameters
        ----------
        node_name : str
            The name of the node.
        param_name : str
            The name of the parameter.

        Returns
        -------
        extended : str
            A name of the form {node_name}.{param_name}
        """
        return f"{node_name}.{param_name}"

    @classmethod
    def from_table(cls, input_table):
        """Create the GraphState from an AstroPy Table with columns for each parameter
        and column names of the form '{node_name}.{param_name}'.

        Parameters
        ----------
        input_table : astropy.table.Table
            The input table.
        """
        num_samples = len(input_table)
        result = GraphState(num_samples=num_samples)
        for col in input_table.colnames:
            components = col.split(".")
            if len(components) != 2:
                raise ValueError(
                    f"Invalid name for entry {col}. Entries should be of the form " f"'node_name.param_name'."
                )

            # If we only have a single value then store that value instead of the np array.
            if num_samples == 1:
                result.set(components[0], components[1], input_table[col].data[0])
            else:
                result.set(components[0], components[1], input_table[col].data)
        return result

    @classmethod
    def from_file(cls, filename):
        """Create the GraphState from a saved file.

        Parameters
        ----------
        filename : str or Path
            The name of the file.
        """
        data_table = ascii.read(filename, format="ecsv")
        return GraphState.from_table(data_table)

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

    def set(self, node_name, var_name, value, force_copy=False, fixed=False):
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
        fixed : `bool`
            Treat this parameter as fixed and do not change it during subsequent calls to set.
            Default: ``False``
        """
        # Check that the names do not use the separator value.
        if "." in node_name or "." in var_name:
            raise ValueError("Names cannot contain the character '.'")

        # Update the meta data.
        if node_name not in self.states:
            self.states[node_name] = {}
            self.fixed_vars[node_name] = set()
        if var_name not in self.states[node_name]:
            self.num_parameters += 1

        # Check if this parameter is fixed. If so, skip the set.
        if var_name in self.fixed_vars[node_name]:
            return

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

        # Mark the variable as fixed if needed.
        if fixed:
            self.fixed_vars[node_name].add(var_name)

    def update(self, inputs, force_copy=False, all_fixed=False):
        """Set multiple parameters' value in the GraphState from a GraphState or a
        dictionary of the same form.

        Note
        ----
        The number of samples in input must either match the number of samples in the
        current object or be 1.

        Parameters
        ----------
        inputs : `GraphState` or `dict`
            Values to copy.
        force_copy : `bool`
            Make a copy of data in an array. If set to ``False`` this will link
            to the array, saving memory and computation time.
            Default: ``False``
        all_fixed : `bool`
            Treat all the parameters in inputs as fixed.
            Default: ``False``

        Raises
        ------
        ValueError if the input an invalid number of samples.
        """
        if isinstance(inputs, GraphState):
            if self.num_samples != inputs.num_samples and inputs.num_samples != 1:
                raise ValueError("GraphSates must have the same number of samples.")
            new_states = inputs.states
        else:
            new_states = inputs

        # Set the values one by one. The set function takes care of expanding
        # any values that are constants (e.g. float or int) to match the correct
        # number of samples.
        for node_name, node_vars in new_states.items():
            for var_name, value in node_vars.items():
                self.set(node_name, var_name, value, force_copy=force_copy, fixed=all_fixed)

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

    def extract_parameters(self, params):
        """Extract the parameter value(s) by a given name. This is often used for
        recording the important parameters from an entire model (set of nodes).

        Parameters
        ----------
        params : str or list-like, optional
            The parameter names to extract. These can be full names ("node.param") or
            use the parameter names.

        Returns
        -------
        values : dict
            The resulting dictionary.
        """
        # If we are looking up a single parameter, but it into a list.
        if isinstance(params, str):
            params = [params]

        # Go through all the parameters. If a parameters full name is provided,
        # look it up now and save the result. Otherwise put it into a list to check
        # for in each node.
        single_params = set()
        results = {}
        for current in params:
            if "." in current:
                node_name, param_name = current.split(".")
                if node_name in self.states and param_name in self.states[node_name]:
                    results[current] = self.states[node_name][param_name]
                else:
                    raise KeyError(f"Parameter {current} not found in GraphState.")
            else:
                single_params.add(current)

        if len(single_params) == 0:
            # Nothing else to do.
            return results

        # Traverse the nested dictionaries looking for cases where the parameter names match.
        first_seen_node = {}
        for node_name, node_params in self.states.items():
            for param_name, param_value in node_params.items():
                if param_name in single_params:
                    if param_name in first_seen_node:
                        # We've already seen this parameter in another node. Time to use the
                        # expanded names.

                        # Start by expanding the result we have already seen if needed.
                        if param_name in results:
                            full_name_existing = f"{first_seen_node[param_name]}.{param_name}"
                            results[full_name_existing] = results[param_name]
                            del results[param_name]

                        # Add the result from the current node.
                        full_name_current = f"{node_name}.{param_name}"
                        results[full_name_current] = param_value
                    else:
                        # This is the first time we have seen the node. Save it with
                        # just the parameter name. Also save the node where we saw it.
                        results[param_name] = param_value
                        first_seen_node[param_name] = node_name

        # Check that we found a match for all the short parameter names.
        for param_name in single_params:
            if param_name not in first_seen_node:
                raise KeyError(f"Parameter {param_name} not found in GraphState.")

        return results

    def to_table(self):
        """Flatten the graph state to an AstroPy Table with columns for each parameter.

        The column names are: {node_name}{separator}{param_name}

        Returns
        -------
        values : astropy.table.Table
            The resulting Table.
        """
        values = Table()
        for node_name, node_params in self.states.items():
            for param_name, param_value in node_params.items():
                values[self.extended_param_name(node_name, param_name)] = np.array(param_value)
        return values

    def to_dict(self):
        """Flatten the graph state to a dictionary with columns for each parameter.

        The column names are: {node_name}{separator}{param_name}

        Returns
        -------
        values : dict
            The resulting dictionary.
        """
        values = {}
        for node_name, node_params in self.states.items():
            for param_name, param_value in node_params.items():
                if self.num_samples == 1:
                    values[self.extended_param_name(node_name, param_name)] = param_value
                else:
                    values[self.extended_param_name(node_name, param_name)] = list(param_value)
        return values

    def save_to_file(self, filename, overwrite=False):
        """Save the GraphState to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save.
        overwrite : bool
            Whether to overwrite an existing file.
            Default: False
        """
        data_table = self.to_table()
        ascii.write(data_table, filename, format="ecsv", overwrite=overwrite)


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
