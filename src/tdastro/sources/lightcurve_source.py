"""A model that generates the SED of a source based on the lightcurves of fluxes
in each band."""

import matplotlib.pyplot as plt
import numpy as np

from tdastro.astro_utils.passbands import Passband, PassbandGroup
from tdastro.consts import lsst_filter_plot_colors
from tdastro.sources.physical_model import PhysicalModel


class LightcurveSource(PhysicalModel):
    """A model that generates the SED of a source from lightcurves in given bands.
    The model estimates a box-shaped SED for each filter such that the resulting
    flux density is equal to the lightcurve's value after passing through
    the passband filter.

    The set of passbands used to configure the model MUST be the same as used
    to generate the SED (the wavelengths must match).

    Parameterized values include:
      * dec - The object's declination in degrees.
      * ra - The object's right ascension in degrees.
      * t0 - The t0 of the zero phase (if applicable), date.

    Attributes
    ----------
    lightcurves : dict
        A dictionary mapping filter names to a 2d array of the bandlfuxes
        in that filter where the first column is time (in days from the reference time
        of the light curve) and the second column is the bandflux (in nJy).
    sed_values : dict
        A dictionary mapping filters to the fake SED values for that passband.
        These SED values are scaled by the lightcurve and added for the
        final SED.
    all_waves : numpy.ndarray
        A 1d array of all of the wavelengths used by the passband group.

    Parameters
    ----------
    lightcurves : dict or numpy.ndarray
        The lightcurves can be passed as either:
        1) a dictionary mapping filter names to a (T, 2) array of the bandlfuxes in that filter
        where the first column is time and the second column is the flux density (in nJy), or
        2) a numpy array of shape (T, 3) array where the first column is time (in days), the
        second column is the bandflux (in nJy), and the third column is the filter.
    passbands : Passband or PassbandGroup
        The passband or passband group to use for defining the lightcurve.
    lc_t0 : float
        The reference epoch (t0) of the input light curve. The model will be shifted
        to the model's t0 when computing fluxes.
        Default: 0.0
    """

    def __init__(
        self,
        lightcurves,
        passbands,
        lc_t0=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Convert a single passband to a PassbandGroup.
        if isinstance(passbands, Passband):
            passbands = PassbandGroup(given_passbands=[passbands])

        # Store the lightcurve information.
        self.lc_t0 = lc_t0
        if isinstance(lightcurves, np.ndarray):
            if lightcurves.shape[1] != 3:
                raise ValueError("Lightcurves must have 3 columns: time, flux, and filter.")

            # Break up the lightcurves by filter and shift so that the time
            # at lc_t0 is mapped to the lightcurve's 0.0 time.
            self.lightcurves = {}
            filters = np.unique(lightcurves[:, 2])
            for filter in filters:
                if filter not in passbands:
                    raise ValueError(f"Lightcurve {filter} does not match any passband in the group.")

                filter_mask = lightcurves[:, 2] == filter
                filter_times = lightcurves[filter_mask, 0].astype(float) - lc_t0
                filter_bandflux = lightcurves[filter_mask, 1].astype(float)
                self.lightcurves[str(filter)] = np.column_stack((filter_times, filter_bandflux))
        elif isinstance(lightcurves, dict):
            self.lightcurves = {}
            for filter, data in lightcurves.items():
                # Validate the dictionary entry.
                if filter not in passbands:
                    raise ValueError(f"Lightcurve {filter} does not match any passband in the group.")
                if len(data.shape) != 2:
                    raise ValueError(f"Lightcurve {filter} must be a 2D array.")
                if data.shape[1] != 2 and data.shape[1] != 3:
                    raise ValueError(f"Lightcurve {filter} must have either 2 or 3 columns.")

                # Copy the lightcurve data so we can shift the times to
                # account for the light_curve's lc_t0.
                self.lightcurves[filter] = np.copy(data)
                self.lightcurves[filter][:, 0] -= lc_t0
        else:
            raise TypeError("Unknown type for lightcurve input. Must be dict or numpy array.")

        # Store the wavelengths information and lightcurves for each filter.
        self.all_waves = passbands.waves
        self.sed_values = self._create_sed_basis(list(self.lightcurves.keys()), passbands)

        # Override some of the defaults of PhysicalModel. Never apply redshift and
        # do not allow brackground models.
        self.apply_redshift = False
        if "background" in kwargs:
            raise ValueError("Lightcurve models do not support background models.")
        self.background = None

        # Check that t0 is set.
        if "t0" not in kwargs or kwargs["t0"] is None:
            raise ValueError("Lightcurve models require a t0 parameter.")

    def _create_sed_basis(self, filters, passbands):
        """Create the SED basis functions. For each passband this creates a box shaped SED
        that does not overlap with any other passband. The height of the SED is normalized
        such that the total flux density will be 1.0 after passing through the passband.

        Parameters
        ----------
        filters : list
            A list of filters to use for the model.
        passbands : PassbandGroup
            The passband group to use for defining the lightcurve.

        Returns
        -------
        sed_values : dict
            A dictionary mapping the filter names to the SED values over all wavelengths.
        """
        # Mark which wavelengths are used by each passband.
        filter_uses_wave = np.zeros((len(filters), len(passbands.waves)))
        for idx, filter in enumerate(filters):
            # Get all of the wavelengths that have a non-negligible transmission value
            # for this filter and find their indices in the passband group.
            is_significant = passbands[filter].processed_transmission_table[:, 1] > 0.001
            significant_waves = passbands[filter].waves[is_significant]
            indices = np.searchsorted(passbands.waves, significant_waves)

            # Mark all non-negligible wavelengths as used by this filter.
            filter_uses_wave[idx, indices] = 1.0

        # Find which wavelengths are used by multiple filters.
        filter_counts = np.sum(filter_uses_wave, axis=0)

        # Create the sed values for each wavelength.
        sed_values = {}
        for idx, filter in enumerate(filters):
            # Get the wavelengths that are used by ONLY this filter.
            valid_waves = (filter_uses_wave[idx, :] == 1) & (filter_counts == 1)
            if np.sum(valid_waves) == 0:
                raise ValueError(
                    f"Passband {filter} has no valid wavelengths where it: a) has a non-negligible "
                    "transmission value (>0.001) and b) does not overlap with another passband."
                )

            # Compute how much flux is passed through these wavelengths of this filter
            # and use this to normalize the sed values.
            filter_sed = np.zeros((1, len(passbands.waves)))
            filter_sed[0, valid_waves] = 1.0

            total_flux = passbands.fluxes_to_bandflux(filter_sed, filter)
            if total_flux[0] <= 0:
                raise ValueError(f"Total flux for filter {filter} is {total_flux[0]}.")
            sed_values[filter] = filter_sed[0, :] / total_flux[0]

        return sed_values

    def set_apply_redshift(self, apply_redshift):
        """Toggles the apply_redshift setting.

        Parameters
        ----------
        apply_redshift : bool
            The new value for apply_redshift.
        """
        raise NotImplementedError("Lightcurve models do not support apply_redshift.")

    def compute_flux(self, times, wavelengths, graph_state):
        """Draw effect-free rest frame flux densities.
        The rest-frame flux is defined as F_nu = L_nu / 4*pi*D_L**2,
        where D_L is the luminosity distance.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps in MJD.
        wavelengths : numpy.ndarray, optional
            A length N array of rest frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of rest frame SED values (in nJy).
        """
        params = self.get_local_params(graph_state)

        # Shift the times for the model's t0 aligned with the lightcurve's lc_t0.
        # The lightcurve times were a;ready shifted in the constructor to be relative to lc_t0.
        shifted_times = times - params["t0"]

        flux_density = np.zeros((len(times), len(wavelengths)))
        for filter, lightcurve in self.lightcurves.items():
            # Compute the SED values for the wavelengths we are actually sampling.
            sed_waves = np.interp(
                wavelengths,  # The query wavelengths
                self.all_waves,  # All of the passband group's wavelengths
                self.sed_values[filter],  # The SED values at each of the passband group's wavelengths
                left=0.0,  # Do not extrapolate in wavelength
                right=0.0,  # Do not extrapolate in wavelength
            )

            # Compute the multipliers for the SEDs at different time steps along this lightcurve.
            sed_time_mult = np.interp(
                shifted_times,  # The query times
                lightcurve[:, 0],  # The lightcurve times for this passband filter
                lightcurve[:, 1],  # The lightcurve flux densities for this passband filter
                left=0.0,  # Do not extrapolate in time
                right=0.0,  # Do not extrapolate in time
            )

            # The contribution of this filter to the overall SED is the lightcurve's (interpolated)
            # value at each time multiplied by the SED values at each query wavelength.
            sed_flux = np.outer(sed_time_mult, sed_waves)
            flux_density += sed_flux

        # Return the total flux density from all lightcurves.
        return flux_density

    def plot_sed_basis(self, ax=None, figure=None):
        """Plot the basis functions for the SED.  This is a debugging
        function to help the user understand the SEDs produced by this
        model.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes or None, optional
            Axes, None by default.
        figure : matplotlib.pyplot.Figure or None
            Figure, None by default.
        """
        if ax is None:
            if figure is None:
                figure = plt.figure()
            ax = figure.add_axes([0, 0, 1, 1])

        # Plot each passband.
        for filter_name, filter_curve in self.sed_values.items():
            color = lsst_filter_plot_colors.get(filter_name, "black")
            ax.plot(self.all_waves, filter_curve, color=color, label=filter_name)

        # Set the x and y axis labels.
        ax.set_xlabel("Wavelength (Angstroms)")
        ax.set_ylabel("SED (nJy)")
        ax.set_title("SED Basis Functions")
        ax.legend()

    def plot_lightcurves(self, times=None, ax=None, figure=None):
        """Plot the underlying lightcurves. This is a debugging
        function to help the user understand the SEDs produced by this
        model.

        Parameters
        ----------
        times : numpy.ndarray or None, optional
            An array of timestamps at which to plot the lightcurves.
            If None, the function uses the timestamps from each lightcurves.
        ax : matplotlib.pyplot.Axes or None, optional
            Axes, None by default.
        figure : matplotlib.pyplot.Figure or None
            Figure, None by default.
        """
        if ax is None:
            if figure is None:
                figure = plt.figure()
            ax = figure.add_axes([0, 0, 1, 1])

        # Plot each passband.
        for filter_name, filter_curve in self.lightcurves.items():
            # Check if we need to use the query times.
            if times is None:
                plot_times = filter_curve[:, 0]
                plot_values = filter_curve[:, 1]
            else:
                plot_times = times
                plot_values = np.interp(times, filter_curve[:, 0], filter_curve[:, 1], left=0.0, right=0.0)

            color = lsst_filter_plot_colors.get(filter_name, "black")
            ax.plot(plot_times, plot_values, color=color, label=filter_name)

        # Set the x and y axis labels.
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Filter value (nJy)")
        ax.set_title("Lightcurve Source Underlying Lightcurves")
        ax.legend()
