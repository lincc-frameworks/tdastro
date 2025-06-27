"""The core functions for running the TDAstro simulation."""

import numpy as np
import pandas as pd
from nested_pandas import NestedFrame

from tdastro.astro_utils.noise_model import apply_noise


def simulate_lightcurves(
    source,
    num_samples,
    opsim,
    passbands,
    opsim_save_cols=None,
    param_cols=None,
    apply_obs_mask=False,
    rng=None,
):
    """Generate a number of simulations of the given source.

    Parameters
    ----------
    source : PhysicalModel
        The source to draw from. This may have its own parameters which
        will be randomly sampled with each draw.
    num_samples : int
        The number of samples.
    opsim : OpSim
        The OpSim information for the samples.
    passbands : PassbandGroup
        The passbands to use for generating the bandfluxes.
    apply_obs_mask: boolean
        If True, apply obs_mask to filter interesting indices/times
    opsim_save_cols : list of str, optional
        A list of opsim columns to be saved as part of the results. This is used
        to save context information about how the lightcurves were generated.
        If None, no additional columns are saved.
    param_cols : list of str, optional
        A list of source parameter columns to be saved as separate columns in
        the results (instead of just the full dictionary of parameters). These
        must be specified as strings in the node_name.param_name format.
        If None, no additional columns are saved.
    rng : numpy.random._generator.Generator, optional
        A given numpy random number generator to use for this computation. If not
        provided, the function uses the node's random number generator.

    Returns
    -------
    lightcurves : nested_pandas.NestedFrame
        A NestedFrame with a row for each object.
    """
    # Sample the parameter space of this model.
    if num_samples <= 0:
        raise ValueError("Invalid number of samples.")
    sample_states = source.sample_parameters(num_samples=num_samples, rng_info=rng)

    # Determine which of the of the simulated positions match opsim locations.
    ra = source.get_param(sample_states, "ra")
    dec = source.get_param(sample_states, "dec")
    all_obs_matches = opsim.range_search(ra, dec)

    # Get all times and all filters as numpy arrays so we can do easy subsets.
    all_times = np.asarray(opsim["time"].values, dtype=float)
    all_filters = np.asarray(opsim["filter"].values, dtype=str)

    # Create dictionaries for keeping all the result information. The first
    # stores per-object information and the second per-object, per-observation.
    # nested_index maps the entry in the nested array to the object's index.
    results_dict = {
        "id": [],
        "ra": [],
        "dec": [],
        "nobs": [],
        "z": [],
        "params": [],
    }
    nested_dict = {
        "mjd": [],
        "filter": [],
        "flux": [],
        "fluxerr": [],
        "flux_perfect": [],
    }
    nested_index = []

    # Add the extra columns to both the results and nested dictionaries.
    if opsim_save_cols is None:
        opsim_save_cols = []
    for col in opsim_save_cols:
        nested_dict[col] = []
    if param_cols is None:
        param_cols = []
    for col in param_cols:
        results_dict[col.replace(".", "_")] = []

    for idx, state in enumerate(sample_states):
        # Find the indices and times where the current source is seen.
        obs_index = np.asarray(all_obs_matches[idx])
        if len(obs_index) == 0:
            obs_times = []
            obs_filters = []
        else:
            obs_times = all_times[obs_index]

            # Filter to only the "interesting" indices / times for this object.
            if apply_obs_mask:
                obs_mask = source.mask_by_time(obs_times, state)
                obs_index = obs_index[obs_mask]
                obs_times = obs_times[obs_mask]
            # Extract the filters for this observation.
            obs_filters = all_filters[obs_index]

        # Compute the band_fluxes and errors over just the given filters.
        bandfluxes_perfect = source.get_band_fluxes(passbands, obs_times, obs_filters, state)
        bandfluxes_error = opsim.bandflux_error_point_source(bandfluxes_perfect, obs_index)
        bandfluxes = apply_noise(bandfluxes_perfect, bandfluxes_error, rng=rng)

        # Save the object level information.
        results_dict["id"].append(idx)
        results_dict["ra"].append(ra[idx])
        results_dict["dec"].append(dec[idx])
        results_dict["nobs"].append(len(obs_times))
        results_dict["z"].append(source.get_param(state, "redshift"))
        results_dict["params"].append(state.to_dict())

        # Save the per-object parameters as separate columns. We use {node_name}_{param_name}
        # as the column name for each parameter since the . notation is used for nested columns.
        for col in param_cols:
            results_dict[col.replace(".", "_")].append(state[col])

        # Append the per-observation data to the nested dictionary, including
        # and needed opsim columns.
        nested_dict["mjd"].extend(list(obs_times))
        nested_dict["filter"].extend(list(obs_filters))
        nested_dict["flux_perfect"].extend(list(bandfluxes_perfect))
        nested_dict["flux"].extend(list(bandfluxes))
        nested_dict["fluxerr"].extend(list(bandfluxes_error))
        for col in opsim_save_cols:
            nested_dict[col].extend(list(opsim[col].values[obs_index]))

        nested_index.extend([idx] * len(obs_times))

    # Create the nested frame.
    results = NestedFrame(data=results_dict, index=[i for i in range(num_samples)])
    nested_frame = pd.DataFrame(data=nested_dict, index=nested_index)
    results = results.add_nested(nested_frame, "lightcurve")
    return results
