"""Utility functions for processing the results data."""


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


def results_compute_snr(results):
    """Compute the signal-to-noise ratio (SNR) for each lightcurve.

    Parameters
    ----------
    results : nested_pandas.NestedFrame
        The DataFrame containing lightcurve data.

    Returns
    -------
    nested_pandas.NestedFrame
        The DataFrame with SNR computed for each lightcurve.
    """
    if "lightcurve" not in results.columns:
        raise ValueError("lightcurve must be present in the DataFrame.")
    if "flux" not in results["lightcurve"].nest.fields:
        raise ValueError("lightcurve.flux must be present in the DataFrame.")
    if "fluxerr" not in results["lightcurve"].nest.fields:
        raise ValueError("lightcurve.fluxerr must be present in the DataFrame.")

    results["lightcurve.snr"] = results["lightcurve.flux"] / results["lightcurve.fluxerr"]
    return results


def results_filter_on_snr(results, snr_threshold=5):
    """Filter results based on a minimum SNR threshold.

    Parameters
    ----------
    results : nested_pandas.NestedFrame
        The DataFrame containing lightcurve data.
    snr_threshold : float, optional
        The minimum SNR threshold for filtering.
        Default: 5.

    Returns
    -------
    nested_pandas.NestedFrame
        The DataFrame with lightcurves filtered by SNR.
    """
    # For safety, we ensure the snr_threshold is a float or integer.
    if not isinstance(snr_threshold, float | int):
        raise ValueError("snr_threshold must be a float or integer.")

    # Add the SNR information if it is not already present.
    if "snr" not in results["lightcurve"].nest.fields:
        results = results_compute_snr(results)

    # Filter the results based on the SNR threshold and drop empty lightcurves.
    results = results.query(f"lightcurve.snr > {snr_threshold}")
    return results.dropna(subset=["lightcurve"])
