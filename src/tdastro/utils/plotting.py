"""A collection of functions for plotting and visualization."""

import matplotlib.pyplot as plt
import numpy as np


def plot_lightcurves(
    fluxes, times, fluxerrs=None, filters=None, ax=None, figure=None, title=None, colormap=None
):
    """Plot one or more lightcurves.

    Parameters
    ----------
    fluxes : numpy.ndarray
        An array of T flux values.
    times : numpy.ndarray
        A length T matrix of the times, used for setting the x axis.
        All times in MJD.
    fluxerrs : numpy.ndarray or None, optional
        A length T matrix of errors on the fluxes for error bars. If not provided
        no error bars are created. None by default.
    filters : numpy.ndarray or None, optional
        A length T matrix of filter names. If not provided all points are
        treated as coming from the same filter. None by default.
    ax : matplotlib.pyplot.Axes or None, optional
        Axes, None by default.
    figure : matplotlib.pyplot.Figure or None
        Figure, None by default.
    title : str or None, optional
        Title of the plot. None by default.
    colormap: dict, optional
        A dictionary that provides mapping between filters and the colors to be plotted.
    """
    # If no axes were given create them using either the given figure or
    # a newly created one (if no figure is given).
    if ax is None:
        if figure is None:
            figure = plt.figure()
        ax = figure.add_axes([0, 0, 1, 1])

    # Set up the time array if it is not given.
    num_pts = len(fluxes)
    if len(times) != num_pts:
        raise ValueError(f"Mismatched array sizes for fluxes ({num_pts}) and times ({len(times)}).")

    # Set up a list of filters to display.
    if filters is None:
        filters = ["none"] * num_pts
        unique_filters = set(["None"])
    elif len(filters) == num_pts:
        filters = np.asarray(filters)
        unique_filters = np.unique(filters)
    else:
        raise ValueError(f"Mismatched array sizes for fluxes ({num_pts}) and filters ({len(filters)}).")

    # Check that if flux errors are given, they are the correct size.
    if fluxerrs is not None and len(fluxerrs) != num_pts:
        raise ValueError(f"Mismatched array sizes for fluxes ({num_pts}) and fluxerrs ({len(fluxerrs)}).")

    if colormap is None:
        colormap = {}
        colors = "bgrcmyk"
        for i, f in enumerate(unique_filters):
            colormap[f] = colors[i]

    # Plot the data with one line for each filter.
    for filter in unique_filters:
        filter_mask = filters == filter

        if fluxerrs is None:
            ax.plot(
                times[filter_mask],
                fluxes[filter_mask],
                marker="o",
                label=filter,
                color=colormap[filter],
            )
        else:
            ax.errorbar(
                times[filter_mask],
                fluxes[filter_mask],
                yerr=fluxerrs[filter_mask],
                fmt="o",
                label=filter,
                color=colormap[filter],
            )

    # Set the title and axis labels.
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Time (MJD)")
    ax.set_ylabel("Flux (nJy)")

    # Only include a legend if there are at least two curves.
    if len(unique_filters) > 1:
        ax.legend()


def plot_bandflux_lightcurves(bandflux, times=None, ax=None, figure=None, title=None):
    """Plot one or more lightcurves where each band is observed at each time.
    This is primarily used for visualizing non-sampled data.

    Parameters
    ----------
    bandflux : numpy.ndarray or dict
        Either a single array with the lightcurve or a dictionary mapping
        lightcurve names to the arrays of values.
    times : numpy.ndarray or None, optional
        A length T matrix of the times, used for setting the x axis. If not
        provided, uses equal spaced ticks. None by default.
    ax : matplotlib.pyplot.Axes or None, optional
        Axes, None by default.
    figure : matplotlib.pyplot.Figure or None
        Figure, None by default.
    title : str or None, optional
        Title of the plot. None by default.
    """
    # If no axes were given create them using either the given figure or
    # a newly created one (if no figure is given).
    if ax is None:
        if figure is None:
            figure = plt.figure()
        ax = figure.add_axes([0, 0, 1, 1])

    # Plot the data.
    if isinstance(bandflux, np.ndarray):
        bandflux = {"lightcurve": bandflux}
    for name, curve in bandflux.items():
        if times is None:
            times = np.arange(len(curve))
        ax.plot(times, curve, marker="o", label=name)

    # Set the title and axis labels.
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux")

    # Only include a legend if there are at least two curves.
    if len(bandflux) > 1:
        ax.legend()


def plot_flux_spectrogram(flux_density, times=None, wavelengths=None, ax=None, figure=None, title=None):
    """Plot a spectrogram to visualize the fluxes.

    Parameters
    ----------
    flux_density : numpy.ndarray
        A length T x N matrix of SED values (in nJy), where  T is the number of time steps,
        and N is the number of wavelengths.
    times : numpy.ndarray or None, optional
        A length T matrix of the times, used for setting the x axis. If not
        provided, uses equal spaced ticks. None by default.
    wavelengths : numpy.ndarray or None, optional
        A length N matrix of the times, used for setting the y axis. If not
        provided, uses equal spaced ticks. None by default.
    ax : matplotlib.pyplot.Axes or None, optional
        Axes, None by default.
    figure : matplotlib.pyplot.Figure or None
        Figure, None by default.
    title : str or None, optional
        Title of the plot. None by default.
    """
    # If no axes were given create them using either the given figure or
    # a newly created one (if no figure is given).
    if ax is None:
        if figure is None:
            figure = plt.figure()
        ax = figure.add_axes([0, 0, 1, 1])

    ax.imshow(flux_density.T, cmap="plasma", interpolation="nearest", aspect="auto")

    # Add title, axis labels, and correct ticks
    if title is None:
        ax.set_title("Flux Spectrogram")
    else:
        ax.set_title(title)

    if times is not None:
        ax.set_xlabel("Time (days)")
        ax.set_xticks(np.arange(len(times))[::4], [f"{round(time)}" for time in times][::4])
    if wavelengths is not None:
        ax.set_ylabel("Wavelength (Angstrom)")
        ax.set_yticks(np.arange(len(wavelengths))[::50], [f"{round(wave)}" for wave in wavelengths][::50])

    # Add flux labels
    for (j, i), label in np.ndenumerate(flux_density.T):
        if i % 2 == 1 and j % 40 == 20:
            ax.text(i, j, round(label, 1), ha="center", va="center", size=8)
