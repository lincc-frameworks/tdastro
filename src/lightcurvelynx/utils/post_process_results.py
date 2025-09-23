"""Utility functions for post processing the results data by adding statistics
columns and filtering on those columns."""

import numpy as np
from nested_pandas import NestedFrame

from lightcurvelynx.astro_utils.mag_flux import flux2mag


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


def results_augment_lightcurves(results, *, min_snr=0.0):
    """Add columns to the results DataFrame with additional information
    about each light curve, including SNR, detection flag, AB magnitude, and
    AB magnitude error.

    The input data frame can either be a single light curve (pandas.DataFrame)
    with columns "flux" and "fluxerr", or a NestedFrame (nested_pandas.NestedFrame)
    with a nested DataFrame column "lightcurve" that contains the "flux" and
    "fluxerr" columns.

    Parameters
    ----------
    results : pandas.DataFrame or nested_pandas.NestedFrame
        The DataFrame containing lightcurve data.
    min_snr : float, optional
        Minimum SNR required to mark an entry as a detection, by default 0.0

    Returns
    -------
    nested_pandas.NestedFrame
        The DataFrame with augmented lightcurve information.
    """
    if not isinstance(results, NestedFrame) or "lightcurve" not in results.columns:
        # Single lightcurve case.
        if "flux" not in results.columns or "fluxerr" not in results.columns:
            raise ValueError("flux and fluxerr must be present in the DataFrame.")
        flux = results["flux"]
        fluxerr = results["fluxerr"]
        prefix = ""
    else:
        if (
            "flux" not in results["lightcurve"].nest.fields
            or "fluxerr" not in results["lightcurve"].nest.fields
        ):
            raise ValueError("lightcurve.flux and lightcurve.fluxerr must be present in the DataFrame.")
        flux = results["lightcurve.flux"]
        fluxerr = results["lightcurve.fluxerr"]
        prefix = "lightcurve."

    # Compute the signal-to-noise ratio (SNR) for each lightcurve entry and whether
    # each entry would be a detection based on the min_snr threshold.
    results[f"{prefix}snr"] = flux / fluxerr
    if min_snr > 0:
        results[f"{prefix}detection"] = results[f"{prefix}snr"] > min_snr

    # Compute the magnitude and magnitude error for each lightcurve entry.
    results[f"{prefix}mag"] = flux2mag(flux)
    results[f"{prefix}magerr"] = (2.5 / np.log(10)) * (fluxerr / flux)

    return results
