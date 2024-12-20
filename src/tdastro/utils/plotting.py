"""A collection of functions for plotting and visualization."""

import matplotlib.pyplot as plt
import numpy as np


def plot_lightcurves(bandflux, times=None, ax=None, figure=None, title=None):
    """Plot one or more lightcurves.

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
