"""A model that generates the SED of a source based on the lightcurves of fluxes
in each band."""

import matplotlib.pyplot as plt
import numpy as np

from tdastro.astro_utils.fake_sed_basis import FakeSEDBasis
from tdastro.astro_utils.passbands import Passband, PassbandGroup
from tdastro.consts import lsst_filter_plot_colors
from tdastro.sources.physical_model import PhysicalModel


class LightcurveSource(PhysicalModel):
    """A model that generates the SED of a source from lightcurves in given bands.
    The model estimates a box-shaped SED for each filter such that the resulting
    flux density is equal to the lightcurve's value after passing through
    the passband filter.

    LightcurveSource supports both periodic and non-periodic lightcurves. If the
    light curve is not periodic then each lightcurve's given values will be interpolated
    during the time range of the lightcurve. Values outside the time range (before and
    after) will be set to the baseline value for that filter (0.0 by default).

    Periodic models require that each filter's lightcurve is sampled at the same times
    and that the value at the end of the lightcurve is equal to the value at the start
    of the lightcurve. The lightcurve epoch (lc_t0) is automatically set to the first time
    so that the t0 parameter corresponds to the shift in phase.

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
    sed_values : FakeSEDBasis
        Fake basis functions for the SED in each filter.
    all_waves : numpy.ndarray
        A 1d array of all of the wavelengths used by the passband group.
    period : float or None
        The period of the lightcurve in days. If the lightcurve is not periodic,
        then this value is set to None.
    min_times : dict
        A dictionary mapping filter names to the minimum time of the lightcurve
        in that filter (relative to lc_t0).
    max_times : dict
        A dictionary mapping filter names to the maximum time of the lightcurve
        in that filter (relative to lc_t0).
    baseline : dict
        A dictionary of baseline bandfluxes for each filter. This is only used
        for non-periodic lightcurves when they are not active.

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
        to the model's t0 when computing fluxes.  For periodic lightcurves, this either
        must be set to the first time of the lightcurve or left as 0.0 to automatically
        derive the lc_t0 from the lightcurve.
        Default: 0.0
    periodic : bool
        Whether the lightcurve is periodic. If True, the model will assume that
        the lightcurve repeats every period.
        Default: False
    baseline : dict or None
        A dictionary of baseline bandfluxes for each filter. This is only used
        for non-periodic lightcurves when they are not active.
        Default: None
    """

    def __init__(
        self,
        lightcurves,
        passbands,
        lc_t0=0.0,
        periodic=False,
        baseline=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Set model information.
        self.period = None
        self.lc_t0 = lc_t0

        # Convert a single passband to a PassbandGroup.
        if isinstance(passbands, Passband):
            passbands = PassbandGroup(given_passbands=[passbands])

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

        # Validate that all the times for each lightcurve are in sorted order.
        for filter, lc in self.lightcurves.items():
            if not np.all(np.diff(lc[:, 0]) > 0):
                raise ValueError(f"Lightcurve {filter}'s times are not in sorted order.")

        # Store the wavelengths information and lightcurves for each filter that we actually use.
        filters = list(self.lightcurves.keys())
        self.sed_values = FakeSEDBasis.from_passbands(passbands, filters=filters)
        self.all_waves = passbands.waves

        # Store information about the lightcurve's periodicity and duration.  We compute
        # the minimum and maximum times for each lightcurve, after _handle_periodicity
        # in case we needed to adjust the lightcurves for periodicity.
        if periodic:
            self._validate_periodicity()
        self.min_times = {filter: lc[0, 0] for filter, lc in self.lightcurves.items()}
        self.max_times = {filter: lc[-1, 0] for filter, lc in self.lightcurves.items()}

        # Store the baseline values for each filter. If the baseline is provided,
        # make sure it contains all of the filters. If no baseline is provided,
        # set the baseline to 0.0 for each filter.
        if baseline is None:
            self.baseline = {filter: 0.0 for filter in self.lightcurves}
        else:
            for filter in self.lightcurves:
                if filter not in baseline:
                    raise ValueError(f"Baseline value for filter {filter} is missing.")
            self.baseline = baseline

        # Override some of the defaults of PhysicalModel. Never apply redshift and
        # do not allow brackground models.
        self.apply_redshift = False
        if "background" in kwargs:
            raise ValueError("Lightcurve models do not support background models.")
        self.background = None

        # Check that t0 is set.
        if "t0" not in kwargs or kwargs["t0"] is None:
            raise ValueError("Lightcurve models require a t0 parameter.")

    def _validate_periodicity(self):
        """Check that the lightcurves meet the requirements for periodic models:
        - All lightcurves must be sampled at the same times.
        - The lightcurves must have a non-zero period.
        - The value at the start and end of each lightcurve must be the same.
        """
        all_lcs = list(self.lightcurves.values())
        if len(all_lcs) == 0:
            raise ValueError("Periodic lightcurve models must have at least one lightcurve.")
        if len(all_lcs[0]) < 2:
            raise ValueError("All periodic lightcurves must have at least two time points.")

        # Check that all lightcurves are sampled at the same times and the first value
        # matches the last value.
        num_curves = len(all_lcs)
        for i in range(num_curves):
            if not np.allclose(all_lcs[i][:, 0], all_lcs[0][:, 0]):
                raise ValueError("All lightcurves in a periodic model must be sampled at the same times.")
            if not np.allclose(all_lcs[i][0, 1], all_lcs[i][-1, 1]):
                raise ValueError("All periodic lightcurves must have the same value at the start and end.")

        # Check that all lightcurves have a non-zero period.
        self.period = all_lcs[0][-1, 0] - all_lcs[0][0, 0]
        if self.period <= 0.0:
            raise ValueError("The period of the lightcurve must be positive.")

        # Shift all the lightcurves so they start at 0 (to make the math easier)
        # and record the offset as lc_t0.
        if not np.isclose(all_lcs[0][0, 0], 0.0):
            if self.lc_t0 != 0.0:
                raise ValueError(
                    "For periodic models, lc_t0 must either be set to the first time "
                    f"or automatically derived. Found lc_t0={self.lc_t0}."
                )

            self.lc_t0 = all_lcs[0][0, 0]
            for lc in self.lightcurves.values():
                lc[:, 0] -= self.lc_t0

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

        # We only support querying wavelengths that were used to create the SED basis
        # functions so as to keep the scaling consistent.  This means we do not allow
        # changes to the wavelengths (such as redshift).
        wave_inds = np.searchsorted(self.all_waves, wavelengths, side="left")
        wave_inds[wave_inds >= len(self.all_waves)] = len(self.all_waves) - 1
        if np.any(np.abs(self.all_waves[wave_inds] - wavelengths) > 0.01):
            raise ValueError(
                "Wavelengths used to query LightcurveSource must be a subset of those used to create it."
            )

        # Shift the times for the model's t0 aligned with the lightcurve's lc_t0.
        # The lightcurve times were already shifted in the constructor to be relative to lc_t0.
        shifted_times = times - params["t0"]
        if self.period is not None:
            shifted_times = shifted_times % self.period

        flux_density = np.zeros((len(times), len(wavelengths)))
        for filter, lightcurve in self.lightcurves.items():
            # Since we have already compute the indices, we can use them directly.
            sed_waves = self.sed_values.sed_values[filter][wave_inds]

            # Compute the multipliers for the SEDs at different time steps along this lightcurve.
            # We use the lightcurve's baseline value for all times outside the lightcurve's range.
            sed_time_mult = np.full(len(shifted_times), self.baseline.get(filter, 0.0))
            overlap_mask = (shifted_times >= self.min_times[filter]) & (
                shifted_times <= self.max_times[filter]
            )

            # For the times that overlap with the lightcurve, interpolate the lightcurve values.
            sed_time_mult[overlap_mask] = np.interp(
                shifted_times[overlap_mask],  # The query times
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
        self.sed_values.plot_sed_basis(ax=ax, figure=figure)

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
