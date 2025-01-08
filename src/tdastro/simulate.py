"""The core functions for running the TDAstro simulation."""

import numpy as np
import pandas as pd
from nested_pandas import NestedFrame

from tdastro.astro_utils.noise_model import apply_noise


def simulate_lightcurves(source, num_samples, opsim, passbands, rng=None):
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

    for idx, state in enumerate(sample_states):
        # Find the indices and times where the current source is seen.
        obs_index = np.asarray(all_obs_matches[idx])
        obs_times = all_times[obs_index]

        # Filter to only the "interesting" indices / times for this object.
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
        results_dict["params"].append(state.to_dict())

        # Append the per-observation data to the nested dictionary.
        nested_dict["mjd"].extend(list(obs_times))
        nested_dict["filter"].extend(list(obs_filters))
        nested_dict["flux_perfect"].extend(list(bandfluxes_perfect))
        nested_dict["flux"].extend(list(bandfluxes))
        nested_dict["fluxerr"].extend(list(bandfluxes_error))
        nested_index.extend([idx] * len(obs_times))

    # Create the nested frame.
    results = NestedFrame(data=results_dict, index=[i for i in range(num_samples)])
    nested_frame = pd.DataFrame(data=nested_dict, index=nested_index)
    results = results.add_nested(nested_frame, "lightcurve")
    return results
