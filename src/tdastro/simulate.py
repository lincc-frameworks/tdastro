"""The core functions for running the TDAstro simulation.
"""

import pandas as pd

from tdastro.astro_utils.noise_model import apply_noise


def simulate_passbands(source, num_samples, opsim, passbands, rng=None):
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
    lightcurves :

    """
    # Sample the parameter space of this model.
    if num_samples <= 0:
        raise ValueError("Invalid number of samples.")
    sample_states = source.sample_parameters(num_samples=num_samples, rng_info=rng)

    # Determine which of the of the simulated positions match opsim locations.
    ra = source.get_param("ra")
    dec = source.get_param("dec")
    all_obs_matches = opsim.range_search(ra, dec)

    # Create a dictionary for keeping all the result information.
    results_dict = {
        "id": [],
        "mjd": [],
        "filter": [],
        "flux": [],
        "fluxerr": [],
    }

    for idx, state in enumerate(sample_states):
        # Find the indices and times where the current source is seen.
        obs_index = all_obs_matches[idx]
        obs_times = opsim["time"][obs_index]

        # Filter to only the "interesting" indices / times for this object.
        obs_mask = source.mask_by_time(obs_times, state)
        obs_index = obs_index[obs_mask]
        obs_times = obs_times[obs_mask]

        # Extract the filters for this observation.
        obs_filters = opsim["filter"][obs_index]

        # Compute the band_fluxes and errors over just the given filters.
        bandfluxes_perfect = source.get_band_fluxes(passbands, obs_times, obs_filters, state)
        bandfluxes_error = opsim.bandflux_error_point_source(bandfluxes_perfect, obs_index)
        bandfluxes = apply_noise(bandfluxes_perfect, bandfluxes_error, rng=rng)

        # Append the data to the results.
        results_dict["id"].append(idx)
        results_dict["mjd"].append(obs_times)
        results_dict["filter"].append(obs_filters)
        results_dict["flux"].append(bandfluxes)
        results_dict["fluxerr"].append(bandfluxes_error)

    return pd.DataFrame(results_dict)
