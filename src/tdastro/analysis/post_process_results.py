"""Utility functions for post processing the results data by adding statistics
columns and filtering on those columns."""

import numpy as np


def results_drop_empty(results):
    """Drop empty lightcurves from the results DataFrame.

    Parameters
    ----------
    results : nested_pandas.NestedFrame
        The DataFrame containing lightcurve data.

    Returns
    -------
    nested_pandas.NestedFrame
        The DataFrame with empty lightcurves removed.
    """
    return results.dropna(subset=["lightcurve"])


def results_append_num_obs(results, min_obs=0):
    """Add a column with the number of observations for each lightcurve by
    counting unique MJD values.

    Parameters
    ----------
    results : nested_pandas.NestedFrame
        The DataFrame containing lightcurve data.
    min_obs : int, optional
        Minimum number of observations required to keep a lightcurve.
        Lightcurves with fewer observations will be dropped.
        Default: 0 (no filtering).

    Returns
    -------
    nested_pandas.NestedFrame
        The DataFrame with an additional column for the number of observations.
    """

    def _get_nobs(mjd_col):
        return len(np.unique(np.floor(mjd_col)))

    nobs = results.reduce(_get_nobs, "lightcurve.mjd")
    results["num_obs"] = nobs.values

    if min_obs > 0:
        results = results[results["num_obs"] >= min_obs]

    return results


def results_append_num_filters(results, min_filters=0):
    """Add a column with the number of filters for each lightcurve.

    Parameters
    ----------
    results : nested_pandas.NestedFrame
        The DataFrame containing lightcurve data.
    min_filters : int, optional
        Minimum number of filters required to keep a lightcurve.
        Lightcurves with fewer filters will be dropped.
        Default: 0 (no filtering).

    Returns
    -------
    nested_pandas.NestedFrame
        The DataFrame with an additional column for the number of filters.
    """

    def _get_nfilters(filter_col):
        return len(np.unique(filter_col))

    nfilters = results.reduce(_get_nfilters, "lightcurve.filter")
    results["num_filters"] = nfilters.values

    if min_filters > 0:
        results = results[results["num_filters"] >= min_filters]

    return results


def results_append_lightcurve_dt(results, min_dt=0):
    """Add a column with the length of each lightcurve (in days)

    Parameters
    ----------
    results : nested_pandas.NestedFrame
        The DataFrame containing lightcurve data.
    min_dt : float, optional
        Minimum duration (in days) required to keep a lightcurve.
        Lightcurves with shorter durations will be dropped.
        Default: 0.0 (no filtering).

    Returns
    -------
    nested_pandas.NestedFrame
        The DataFrame with an additional column for the lightcurve duration.
    """

    def _get_lightcurve_dt(mjd_col):
        return np.max(mjd_col) - np.min(mjd_col)

    lightcurve_dt = results.reduce(_get_lightcurve_dt, "lightcurve.mjd")
    results["lightcurve_dt"] = lightcurve_dt.values

    if min_dt > 0:
        results = results[results["lightcurve_dt"] >= min_dt]

    return results


def results_append_lightcurve_snr(results, min_snr=0.0):
    """Add a column with the SNR for each lightcurve and (optionally)
    filter lightcurves with SNR below a threshold.

    Parameters
    ----------
    results : nested_pandas.NestedFrame
        The DataFrame containing lightcurve data.
    min_snr : float, optional
        Minimum SNR required to keep an individual entry in a lightcurve.
        Lightcurve entries with lower SNR will be dropped.
        Default: 0.0 (no filtering).

    Returns
    -------
    nested_pandas.NestedFrame
        The DataFrame with SNR computed for each entry in the lightcurve.
    """
    if "lightcurve" not in results.columns:
        raise ValueError("lightcurve must be present in the DataFrame.")
    results = results.dropna(subset=["lightcurve"])

    if "flux" not in results["lightcurve"].nest.fields:
        raise ValueError("lightcurve.flux must be present in the DataFrame.")
    if "fluxerr" not in results["lightcurve"].nest.fields:
        raise ValueError("lightcurve.fluxerr must be present in the DataFrame.")

    results["lightcurve.snr"] = results["lightcurve.flux"] / results["lightcurve.fluxerr"]

    # Filter the results based on the SNR threshold and drop empty lightcurves.
    min_snr = float(min_snr)  # Ensure min_snr is a float
    if min_snr > 0:
        results = results.query(f"lightcurve.snr > {float(min_snr)}")
        results = results.dropna(subset=["lightcurve"])
    return results
