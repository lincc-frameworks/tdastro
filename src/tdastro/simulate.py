"""The core functions for running the TDAstro simulation."""

import citation_compass as cc
import numpy as np
import pandas as pd
from nested_pandas import NestedFrame

from tdastro.astro_utils.noise_model import apply_noise
from tdastro.models.physical_model import BandfluxModel


def get_time_windows(t0, time_window_offset):
    """Get the time windows for each sample state based on the time window offset.

    Parameters
    ----------
    t0 : float or np.ndarray, optional
        The reference time (t0) for the time windows.
    time_window_offset : tuple(float, float), optional
        A tuple specifying the time window offset (before, after) t0 in days.
        If None, no time window is applied.

    Returns
    -------
    start_times : np.ndarray or None
        The start times for each sample t0 - time_window_offset[0]. If a before time is given,
        this is always returned as an array (even if t0 is a scalar). None returned if there is
        no start time.
    end_times : np.ndarray or None
        The end times for each sample t0 - time_window_offset[1]. If an after time is given,
        this is always returned as an array (even if t0 is a scalar). None returned if there is
        no end time.
    """
    # If the model did not have a t0 or we do not have a time_window_offset,
    # we cannot apply a time window.
    if t0 is None or time_window_offset is None:
        return None, None
    if len(time_window_offset) != 2:
        raise ValueError("time_window_offset must be a tuple of (before, after) in days.")
    before, after = time_window_offset

    # If t0 is a scalar apply the offset directly.
    if np.isscalar(t0):
        t0 = np.array([t0])
    start_times = t0 - before if before is not None else None
    end_times = t0 + after if after is not None else None

    return start_times, end_times


def simulate_lightcurves(
    model,
    num_samples,
    obstable,
    passbands,
    *,
    obstable_save_cols=None,
    param_cols=None,
    apply_obs_mask=False,
    time_window_offset=None,
    rng=None,
    generate_citations=False,
):
    """Generate a number of simulations of the given model and information
    from one or more surveys.

    Parameters
    ----------
    model : BasePhysicalModel
        The model to draw from. This may have its own parameters which
        will be randomly sampled with each draw.
    num_samples : int
        The number of samples.
    obstable : ObsTable or List of ObsTable
        The ObsTable(s) from which to extract information for the samples.
    passbands : PassbandGroup or List of PassbandGroup
        The passbands to use for generating the bandfluxes.
    apply_obs_mask: boolean
        If True, apply obs_mask to filter interesting indices/times.
    time_window_offset : tuple(float, float), optional
        A tuple specifying the time window offset (before, after) t0 in days.
        This is used to filter the observations to only those within the specified
        time window (t0 - before, t0 + after). If None or the model does not have a
        t0 specified, no time window is applied.
    obstable_save_cols : list of str, optional
        A list of ObsTable columns to be saved as part of the results. This is used
        to save context information about how the light curves were generated. If the column
        is missing from one of the ObsTables, a null value such as None or NaN is used.
        If None, no additional columns are saved.
    param_cols : list of str, optional
        A list of the model's parameter columns to be saved as separate columns in
        the results (instead of just the full dictionary of parameters). These
        must be specified as strings in the node_name.param_name format.
        If None, no additional columns are saved.
    rng : numpy.random._generator.Generator, optional
        A given numpy random number generator to use for this computation. If not
        provided, the function uses the node's random number generator.
    generate_citations : bool, optional
        If True, generate citations for the simulation and output them.

    Returns
    -------
    lightcurves : nested_pandas.NestedFrame
        A NestedFrame with a row for each object.
    """
    # Sample the parameter space of this model. We do this once for all surveys, so the
    # object use the same parameters across all observations.
    if num_samples <= 0:
        raise ValueError("Invalid number of samples.")
    sample_states = model.sample_parameters(num_samples=num_samples, rng_info=rng)

    # If we are given information for a single survey, make it into a list.
    if not isinstance(obstable, list):
        obstable = [obstable]
    if not isinstance(passbands, list):
        passbands = [passbands]
    num_surveys = len(obstable)
    if num_surveys != len(passbands):
        raise ValueError("Number of surveys must match number of passbands.")

    # We do not currently support bandflux models with multiple surveys because
    # a bandflux model is defined relative to a single survey.
    if num_surveys > 1 and isinstance(model, BandfluxModel):
        raise ValueError(
            "Simulating a BandfluxModel with multiple surveys is currently not supported, "
            "because the bandflux model is defined relative to the filters of a single survey."
        )

    # Create a dictionary for the object level information, including any saved parameters.
    # Some of these are placeholders (e.g. nobs) until they can be filled in during the simulation.
    ra = np.atleast_1d(model.get_param(sample_states, "ra"))
    dec = np.atleast_1d(model.get_param(sample_states, "dec"))
    results_dict = {
        "id": [i for i in range(num_samples)],
        "ra": ra.tolist(),
        "dec": dec.tolist(),
        "nobs": [0] * num_samples,
        "z": np.atleast_1d(model.get_param(sample_states, "redshift")).tolist(),
        "params": [state.to_dict() for state in sample_states],
    }
    if param_cols is not None:
        for col in param_cols:
            if col not in sample_states:
                raise KeyError(f"Parameter column {col} not found in model parameters.")
            results_dict[col.replace(".", "_")] = np.atleast_1d(sample_states[col]).tolist()

    # Set up the nested array for the per-observation data, including ObsTable information.
    nested_index = []
    nested_dict = {"mjd": [], "filter": [], "flux": [], "fluxerr": [], "flux_perfect": [], "survey_idx": []}
    if obstable_save_cols is None:
        obstable_save_cols = []
    for col in obstable_save_cols:
        nested_dict[col] = []

    # Determine which of the of the simulated positions match the pointings from each ObsTable.
    start_times, end_times = get_time_windows(
        model.get_param(sample_states, "t0"),
        time_window_offset,
    )
    all_obs_matches = [
        obstable[i].range_search(ra, dec, t_min=start_times, t_max=end_times) for i in range(num_surveys)
    ]

    # Get all times and all filters as numpy arrays so we can do easy subsets.
    all_times = [np.asarray(obstable[i]["time"].values, dtype=float) for i in range(num_surveys)]
    all_filters = [np.asarray(obstable[i]["filter"].values, dtype=str) for i in range(num_surveys)]

    # We loop over objects first, then surveys. This allows us to generate a single block
    # of data for the object over all surveys.
    for idx, state in enumerate(sample_states):
        total_num_obs = 0

        for survey_idx in range(num_surveys):
            # Find the indices and times where the current model is seen.
            obs_index = np.asarray(all_obs_matches[survey_idx][idx])
            if len(obs_index) == 0:
                obs_times = []
                obs_filters = []
            else:
                obs_times = all_times[survey_idx][obs_index]

                # Filter to only the "interesting" indices / times for this object.
                if apply_obs_mask:
                    obs_mask = model.mask_by_time(obs_times, state)
                    obs_index = obs_index[obs_mask]
                    obs_times = obs_times[obs_mask]
                # Extract the filters for this observation.
                obs_filters = all_filters[survey_idx][obs_index]

            # Compute the bandfluxes and errors over just the given filters.
            bandfluxes_perfect = model.evaluate_bandfluxes(
                passbands[survey_idx], obs_times, obs_filters, state
            )
            bandfluxes_error = obstable[survey_idx].bandflux_error_point_source(bandfluxes_perfect, obs_index)
            bandfluxes = apply_noise(bandfluxes_perfect, bandfluxes_error, rng=rng)

            # Append the per-observation data to the nested dictionary, including
            # and needed ObsTable columns.
            nobs = len(obs_times)
            nested_dict["mjd"].extend(list(obs_times))
            nested_dict["filter"].extend(list(obs_filters))
            nested_dict["flux_perfect"].extend(list(bandfluxes_perfect))
            nested_dict["flux"].extend(list(bandfluxes))
            nested_dict["fluxerr"].extend(list(bandfluxes_error))
            nested_dict["survey_idx"].extend([survey_idx] * nobs)
            for col in obstable_save_cols:
                col_data = (
                    list(obstable[survey_idx][col].values[obs_index])
                    if col in obstable[survey_idx]
                    else [None] * nobs
                )
                nested_dict[col].extend(col_data)

            total_num_obs += nobs
            nested_index.extend([idx] * nobs)

        # The number of observations is the total across all surveys.
        results_dict["nobs"][idx] = total_num_obs

    # Create the nested frame.
    results = NestedFrame(data=results_dict, index=[i for i in range(num_samples)])
    nested_frame = pd.DataFrame(data=nested_dict, index=nested_index)
    results = results.add_nested(nested_frame, "lightcurve")

    # If requested, generate citations for the simulation.
    if generate_citations:
        print(
            "The following citations were called during this simulation. Note that this list is "
            "provided for convenience and may be incomplete and we recommend the user confirm "
            "which models, effects, and parameters were used."
        )
        cc.print_used_citations()

    return results
