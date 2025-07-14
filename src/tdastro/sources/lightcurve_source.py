"""Model that generate the SED or bandflux of a source based on given observer frame
lightcurves of fluxes in each band.

If we are generating the bandfluxes directly, the models interpolate the given lightcurves
at the requested times and filters. If we are generating an SED for a given set of
wavelengths, the model computes a box-shaped SED basis function for each filter that
will produce the same bandflux after being passed through the passband filter.
"""

import logging
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np

from tdastro.astro_utils.mag_flux import mag2flux
from tdastro.astro_utils.passbands import Passband, PassbandGroup
from tdastro.consts import lsst_filter_plot_colors
from tdastro.math_nodes.given_sampler import GivenValueSampler, GivenValueSelector
from tdastro.sources.physical_model import PhysicalModel
from tdastro.utils.io_utils import read_lclib_data

logger = logging.getLogger(__name__)


class LightcurveData:
    """A class to hold data for a single model lightcurve (set of fluxes over time for
    each filter).

    Attributes
    ----------
    lightcurves : dict
        A dictionary mapping filter names to a 2D array of the bandfluxes in that filter,
        where the first column is time (in days from the reference time of the light curve)
        and the second column is the bandflux (in nJy).
    lc_t0 : float
        The reference epoch (t0) of the input light curve. The model will be shifted
        to the model's t0 when computing fluxes. For periodic lightcurves, this either
        must be set to the first time of the lightcurve or left as 0.0 to automatically
        derive the lc_t0 from the lightcurve.
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

    def __init__(self, lightcurves, lc_t0=0.0, periodic=False, baseline=None):
        self.lc_t0 = lc_t0
        self.period = None

        if isinstance(lightcurves, dict):
            # Make a copy of the lightcurves to avoid modifying the original data.
            self.lightcurves = {filter: lc.copy() for filter, lc in lightcurves.items()}
        elif isinstance(lightcurves, np.ndarray):
            if lightcurves.shape[1] != 3:
                raise ValueError("Lightcurves array must have 3 columns: time, flux, and filter.")

            # Break up the lightcurves by filter.
            self.lightcurves = {}
            filters = np.unique(lightcurves[:, 2])
            for filter in filters:
                filter_mask = lightcurves[:, 2] == filter
                filter_times = lightcurves[filter_mask, 0].astype(float)
                filter_bandflux = lightcurves[filter_mask, 1].astype(float)
                self.lightcurves[str(filter)] = np.column_stack((filter_times, filter_bandflux))
        else:
            raise TypeError("Unknown type for lightcurve input. Must be dict, numpy array, or astropy Table.")

        # Do basic validation of the lightcurves and shift them so that the time
        # at lc_t0 is mapped to 0.0.
        for filter, lc in self.lightcurves.items():
            if len(lc.shape) != 2 or (lc.shape[1] != 2 and lc.shape[1] != 3):
                raise ValueError(f"Lightcurve {filter} must have either 2 or 3 columns.")
            if not np.all(np.diff(lc[:, 0]) > 0):
                raise ValueError(f"Lightcurve {filter}'s times are not in sorted order.")
            lc[:, 0] -= self.lc_t0

        # Store the minimum and maximum times for each lightcurve. This is done after
        # validating periodicity in case we needed to adjust the lightcurve start times.
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

    def __len__(self):
        """Get the number of lightcurves."""
        return len(self.lightcurves)

    @property
    def filters(self):
        """Get the list of filters in the lightcurves."""
        return list(self.lightcurves.keys())

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

    @classmethod
    def from_lclib_table(cls, lightcurves_table, lc_t0=0.0, filters=None):
        """Break up a lightcurves table in LCLIB format into a LightcurveData instance.
        This function expects the table to have a "time" column, an optional "type" column,
        and a column for each filter. The "type" column should use "S" for source observation
        and "T" for template (background) observation.

        Parameters
        ----------
        lightcurves_table : astropy.table.Table
            A table with a "time" column, optional "type" column, and a column for each filter.
            If the type column is present it should use "S" for source observation and "T"
            for template (background) observation.
        lc_t0 : float
            The reference epoch (t0) of the input light curve. The model will be shifted
            to the model's t0 when computing fluxes.  For periodic lightcurves, this either
            must be set to the first time of the lightcurve or left as 0.0 to automatically
            derive the lc_t0 from the lightcurve.
            Default: 0.0
        filters : list of str or None
            A list of filters to use for the light curves. If None, all filters will be used.
            Used to select a subset of filters.
            Default: None
        """
        if "time" not in lightcurves_table.colnames:
            raise ValueError("Lightcurves table must have a 'time' column.")

        # Extract the name of the filters from the table column names.
        filter_cols = [col for col in lightcurves_table.colnames if col != "time" and col != "type"]
        if filters is None:
            filters = filter_cols
        else:
            to_keep = set(filter_cols) & set(filters)
            filters = list(to_keep)
        if len(filters) == 0:
            raise ValueError("Lightcurves table must have at least one filter column.")

        # Check if there are baseline curves to extract and filter them out of the
        # lightcurves table. Use a default to 0.0 for each filter if no baselines are found.
        baseline = {filter: 0.0 for filter in filters}
        if "type" in lightcurves_table.colnames:
            obs_mask = lightcurves_table["type"] == "S"
            if np.any(~obs_mask):
                tmp_table = lightcurves_table[~obs_mask]
                if len(tmp_table) > 1:
                    logger.warning(
                        "Multiple template (background) observations found in lightcurves table. "
                        "The source will only use the first one for baseline values."
                    )
                baseline = {filter: tmp_table[filter][0] for filter in filters}
            lightcurves_table = lightcurves_table[obs_mask]

        # Convert the Table to a dictionary of lightcurves.
        lightcurves = {}
        for filter in filters:
            filter_times = lightcurves_table["time"].astype(float)
            filter_bandflux = mag2flux(lightcurves_table[filter].astype(float))
            lightcurves[str(filter)] = np.column_stack((filter_times, filter_bandflux))

        # Check the metadata for periodicity information.
        recur_class = lightcurves_table.meta.get("RECUR_CLASS", "")
        if recur_class == "PERIODIC" or recur_class == "RECUR-PERIODIC":
            periodic = True
            baseline = None  # Baseline is not used for periodic lightcurves.
        elif recur_class == "RECUR-NONPERIODIC":
            periodic = False
            logger.warning("Recurring non-periodic lightcurves are treated as non-recurring within TDAstro.")
        elif recur_class == "NON-RECUR":
            periodic = False
        elif recur_class == "":
            periodic = False
            logger.warning(
                "No RECUR_CLASS metadata found in lightcurves table. Using non-periodic lightcurves."
            )
        else:
            raise ValueError(
                f"Unknown RECUR_CLASS value in lightcurves table metadata: {recur_class}. "
                "Expected 'PERIODIC', 'RECUR-PERIODIC', 'RECUR-NONPERIODIC', or 'NON-RECUR'."
            )

        # If the lightcurves are periodic, make sure they start and end at the same value.
        if periodic:
            all_match = True
            for lc in lightcurves.values():
                all_match &= np.isclose(lc[0, 1], lc[-1, 1])

            # Insert a value to wrap. This should be a bit after the last time
            # and have the same value as the first time.
            if not all_match:
                dt = lightcurves_table["time"][-1] - lightcurves_table["time"][0]
                ave_dt = dt / (len(lightcurves_table) - 1)
                new_end = lightcurves_table["time"][-1] + ave_dt

                for filter, lc in lightcurves.items():
                    lc = np.vstack((lc, [new_end, lc[0, 1]]))
                    lightcurves[filter] = lc

        return cls(lightcurves, lc_t0=lc_t0, periodic=periodic, baseline=baseline)

    def evaluate(self, times, filter):
        """Get the bandflux values for a given filter at the specified times. These can
        be multiplied by a basis SED function to produce estimated SED values
        for the given filter at the specified times or can be used directly as bandfluxes.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of times (in days) at which to compute the SED values. These should
            be shift to be relative to the lightcurve's lc_t0.
        filter : str
            The name of the filter for which to compute the SED values.

        Returns
        -------
        values : numpy.ndarray
            A length T array of bandpass fluxes for the specified filter at the given times.
        """
        if filter not in self.lightcurves:
            raise ValueError(f"Filter {filter} not found in lightcurves.")
        lightcurve = self.lightcurves[filter]

        # If the lightcurve is periodic, wrap the times around the period.
        if self.period is not None:
            times = times % self.period

        # Start with an array of all baseline values.
        values = np.full(len(times), self.baseline.get(filter, 0.0))

        # For the times that overlap with the lightcurve, interpolate the lightcurve values.
        overlap = (times >= self.min_times[filter]) & (times <= self.max_times[filter])
        values[overlap] = np.interp(
            times[overlap],  # The query times
            lightcurve[:, 0],  # The lightcurve times for this passband filter
            lightcurve[:, 1],  # The lightcurve flux densities for this passband filter
            left=0.0,  # Do not extrapolate in time
            right=0.0,  # Do not extrapolate in time
        )

        return values

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


class BaseLightcurveSource(PhysicalModel, ABC):
    """A base class for lightcurve source models. This class is not meant to be used directly,
    but rather as a base for other lightcurve source models that may have additional functionality.
    It provides the basic structure (primarily SED basis functions) and validation for
    lightcurve-based SED models.

    The set of passbands used to configure the model MUST be the same as used
    to generate the SED (the wavelengths must match).

    Parameterized values include:
      * dec - The object's declination in degrees.
      * ra - The object's right ascension in degrees.
      * t0 - The t0 of the zero phase (if applicable), date.

    Attributes
    ----------
    sed_values : dict
        A dictionary mapping filters to the SED basis values for that passband. These SED values can
        be scaled by the lightcurve (bandfluxes) and added together to produce an estimated SED.
    all_waves : numpy.ndarray
        A 1d array of all of the wavelengths used by the passband group.
    filters : list
        A list of all supported filters in the lightcurve model.

    Parameters
    ----------
    passbands : Passband or PassbandGroup
        The passband or passband group to use for defining the lightcurve.
    filters : list
        A list of filter names that the model supports. If None then
        all available filters will be used.
    """

    def __init__(self, passbands, *, filters=None, **kwargs):
        super().__init__(**kwargs)

        # Convert a single passband to a PassbandGroup.
        if isinstance(passbands, Passband):
            passbands = PassbandGroup(given_passbands=[passbands])

        # Check that we have passbands for each filter.
        if filters is None:
            filters = passbands.filters
        else:
            for filter in filters:
                if filter not in passbands:
                    raise ValueError(f"Lightcurve model requires a passband for filter {filter}.")
        self.filters = filters

        self.all_waves = passbands.waves
        self.sed_values = self._create_sed_basis(self.filters, passbands)

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
        sed_basis_values : dict
            A dictionary mapping the filter names to the SED basis values over all wavelengths.
        """
        # Mark which wavelengths are used by each passband.
        waves_per_filter = np.zeros((len(filters), len(passbands.waves)))
        for idx, filter in enumerate(filters):
            # Get all of the wavelengths that have a non-negligible transmission value
            # for this filter and find their indices in the passband group.
            is_significant = passbands[filter].processed_transmission_table[:, 1] > 1e-5
            significant_waves = passbands[filter].waves[is_significant]
            indices = np.searchsorted(passbands.waves, significant_waves)

            # Mark all non-negligible wavelengths as used by this filter.
            waves_per_filter[idx, indices] = 1.0

        # Find which wavelengths are used by multiple filters.
        filter_counts = np.sum(waves_per_filter, axis=0)

        # Create the sed values for each wavelength.
        sed_basis_values = {}
        for idx, filter in enumerate(filters):
            # Get the wavelengths that are used by ONLY this filter.
            valid_waves = (waves_per_filter[idx, :] == 1) & (filter_counts == 1)
            if np.sum(valid_waves) == 0:
                raise ValueError(
                    f"Passband {filter} has no valid wavelengths where it: a) has a non-negligible "
                    "transmission value (>0.001) and b) does not overlap with another passband."
                )

            # Compute how much flux is passed through these wavelengths of this filter
            # and use this to normalize the sed values.
            filter_sed_basis = np.zeros((1, len(passbands.waves)))
            filter_sed_basis[0, valid_waves] = 1.0

            total_flux = passbands.fluxes_to_bandflux(filter_sed_basis, filter)
            if total_flux[0] <= 0:
                raise ValueError(f"Total flux for filter {filter} is {total_flux[0]}.")
            sed_basis_values[filter] = filter_sed_basis[0, :] / total_flux[0]

        return sed_basis_values

    def set_apply_redshift(self, apply_redshift):
        """Toggles the apply_redshift setting.

        Parameters
        ----------
        apply_redshift : bool
            The new value for apply_redshift.
        """
        raise NotImplementedError("Lightcurve models do not support apply_redshift.")

    def add_effect(self, effect):
        """Add an effect to the model.

        Parameters
        ----------
        effect : EffectModel
            The effect to add.
        """
        raise NotImplementedError(
            "Lightcurve-based models are defined in the observer frame and do not support effects."
        )

    def compute_flux_given_lc(self, lc, times, wavelengths, graph_state):
        """Compute the flux density for a given lightcurve at specified times and wavelengths.

        Parameters
        ----------
        lc : LightcurveData
            The lightcurve data to use for computing the flux density.
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray, optional
            A length N array of observer frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of observer frame SED values (in nJy).
        """
        params = self.get_local_params(graph_state)

        # Shift the times for the model's t0 aligned with the lightcurve's lc_t0.
        # The lightcurve times were already shifted in the constructor to be relative to lc_t0.
        shifted_times = times - params["t0"]

        flux_density = np.zeros((len(times), len(wavelengths)))
        for filter in lc.filters:
            # Compute the SED values for the wavelengths we are actually sampling.
            sed_waves = np.interp(
                wavelengths,  # The query wavelengths
                self.all_waves,  # All of the passband group's wavelengths
                self.sed_values[filter],  # The SED values at each of the passband group's wavelengths
                left=0.0,  # Do not extrapolate in wavelength
                right=0.0,  # Do not extrapolate in wavelength
            )

            # Compute the multipliers for the SEDs at different time steps along this lightcurve.
            # We use the lightcurve's baseline value for all times outside the lightcurve's range.
            sed_time_mult = lc.evaluate(shifted_times, filter)

            # The contribution of this filter to the overall SED is the lightcurve's (interpolated)
            # value at each time multiplied by the SED values at each query wavelength.
            sed_flux = np.outer(sed_time_mult, sed_waves)
            flux_density += sed_flux

        # Return the total flux density from all lightcurves.
        return flux_density

    def compute_bandflux(self, times, filters, state, rng_info=None):
        """Evaluate the model at the passband level for a single, given graph state.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filters : numpy.ndarray
            A length T array of filter names.
        state : GraphState
            An object mapping graph parameters to their values with num_samples=1.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        """
        raise NotImplementedError

    def get_band_fluxes(self, passband_or_group, times, filters, state, rng_info=None) -> np.ndarray:
        """Get the band fluxes for a given Passband or PassbandGroup.

        Parameters
        ----------
        passband_or_group : Passband or PassbandGroup
            The passband (or passband group) to use.
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filters : numpy.ndarray or None
            A length T array of filter names. It may be None if
            passband_or_group is a Passband.
        state : GraphState
            An object mapping graph parameters to their values.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.

        Returns
        -------
        band_fluxes : numpy.ndarray
            A matrix of the band fluxes. If only one sample is provided in the GraphState,
            then returns a length T array. Otherwise returns a size S x T array where S is the
            number of samples in the graph state.
        """
        if isinstance(passband_or_group, Passband):
            if filters is not None and not np.all(filters == passband_or_group.filter_name):
                raise ValueError(
                    "If passband_or_group is a Passband, filters must either be None "
                    "or a list where every entry matches the given filter's name: "
                    f"{passband_or_group.filter_name}."
                )
            passband_or_group = PassbandGroup(given_passbands=[passband_or_group])

        if filters is None:
            raise ValueError("If passband_or_group is a PassbandGroup, filters must be provided.")
        filters = np.asarray(filters)

        # Check if we need to sample the graph.
        if state is None:
            state = self.sample_parameters(num_samples=1, rng_info=rng_info)

        results = np.empty((state.num_samples, len(times)))
        for sample_num, current_state in enumerate(state):
            # Compute the flux (applying all effects) and save the result.
            results[sample_num, :] = self.compute_bandflux(times, filters, current_state)

        if state.num_samples == 1:
            return results[0, :]
        return results

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


class LightcurveSource(BaseLightcurveSource):
    """A model that generates either the SED or bandflux of a source based on
    given lightcurves in each band. When generating the bandflux, it interpolates
    the lightcurves directly. When generating the SED, the model uses a box-shaped SED
    for each filter such that the resulting flux density is equal to the lightcurve's
    value after passing through the passband filter.

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
    lightcurves : LightcurveData
        The data for the lightcurves, such as the times and bandfluxes in each filter.
    sed_values : dict
        A dictionary mapping filters to the SED basis values for that passband.
        These SED values are scaled by the lightcurve and added for the
        final SED.
    all_waves : numpy.ndarray
        A 1d array of all of the wavelengths used by the passband group.

    Parameters
    ----------
    lightcurves : dict or numpy.ndarray
        The lightcurves can be passed as either:
        1) a LightcurveData instance,
        2) a dictionary mapping filter names to a (T, 2) array of the bandlfuxes in that filter
        where the first column is time and the second column is the flux density (in nJy), or
        3) a numpy array of shape (T, 3) array where the first column is time (in days), the
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
        *,
        lc_t0=0.0,
        periodic=False,
        baseline=None,
        **kwargs,
    ):
        # Store the lightcurve data, parsing out different formats if needed.
        self.lightcurves = LightcurveData(lightcurves, lc_t0=lc_t0, periodic=periodic, baseline=baseline)
        super().__init__(passbands, filters=self.lightcurves.filters, **kwargs)

    def compute_flux(self, times, wavelengths, graph_state):
        """Draw effect-free observer frame flux densities.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray, optional
            A length N array of observer frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of observer frame SED values (in nJy). These are generated
            from non-overlapping box-shaped SED basis functions for each filter and
            scaled by the lightcurve values.
        """
        return self.compute_flux_given_lc(
            self.lightcurves,
            times,
            wavelengths,
            graph_state,
        )

    def compute_bandflux(self, times, filters, state, **kwargs):
        """Evaluate the model at the passband level for a single, given graph state.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filters : numpy.ndarray
            A length T array of filter names.
        state : GraphState
            An object mapping graph parameters to their values with num_samples=1.
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        band_fluxes : numpy.ndarray
            A length T matrix of observer frame passband fluxes (in nJy).
        """
        params = self.get_local_params(state)

        # Check that the filters are all supported by the model.
        for flt in np.unique(filters):
            if flt not in self.lightcurves.lightcurves:
                raise ValueError(f"Filter '{flt}' is not supported by LightcurveSource.")

        # Shift the times for the model's t0 aligned with the lightcurve's lc_t0.
        # The lightcurve times were already shifted in the constructor to be relative to lc_t0.
        shifted_times = times - params["t0"]

        band_fluxes = np.zeros(len(times))
        for filter in self.lightcurves.filters:
            filter_mask = filters == filter
            band_fluxes[filter_mask] = self.lightcurves.evaluate(shifted_times[filter_mask], filter)

        return band_fluxes

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
        self.lightcurves.plot_lightcurves(times=times, ax=ax, figure=figure)


class MultiLightcurveSource(BaseLightcurveSource):
    """A MultiLightcurveSource randomly selects a lightcurve at each evaluation
    computes the flux from that source. The models can generate either the SED or
    bandflux of a source based of given lightcurves in each band. When generating
    the bandflux, the model interpolates the lightcurves directly. When generating the SED,
    the model uses a box-shaped SED for each filter such that the resulting flux density
    is equal to the lightcurve's value after passing through the passband filter.

    MultiLightcurveSource supports both periodic and non-periodic lightcurves. If the
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
    lightcurves : list of LightcurveData
        The data for each set of lightcurves.
    sed_values : dict
        A dictionary mapping filters to the SED basis values for that passband.
        These SED values are scaled by the lightcurve and added for the
        final SED.
    all_waves : numpy.ndarray
        A 1d array of all of the wavelengths used by the passband group.
    all_filters : set
        A set of all filters used by the lightcurves. This is the union of all
        filters used by each lightcurve in the lightcurves list.

    Parameters
    ----------
    lightcurves : list of LightcurveData
        The data for each set of lightcurves. One lightcurve will be randomly selected
        at each evaluation.
    passbands : Passband or PassbandGroup
        The passband or passband group to use for defining the lightcurve.
    weights : numpy.ndarray, optional
        A length N array indicating the relative weight from which to select
        a source at random. If None, all sources will be weighted equally.
    """

    def __init__(
        self,
        lightcurves,
        passbands,
        *,
        weights=None,
        **kwargs,
    ):
        # Validate the lightcurve input and create a union of all filters used.
        self.all_filters = set()
        for lc in lightcurves:
            if not isinstance(lc, LightcurveData):
                raise TypeError("Each lightcurve must be an instance of LightcurveData.")
            self.all_filters.update(lc.filters)
        self.lightcurves = lightcurves

        super().__init__(passbands, filters=list(self.all_filters), **kwargs)

        source_inds = [i for i in range(len(lightcurves))]
        self._sampler_node = GivenValueSampler(source_inds, weights=weights)
        self.add_parameter("selected_lightcurve", value=self._sampler_node, allow_gradient=False)

        # Assemble a list of baseline values for each filter across all lightcurves.
        # Create a parameter to track the baseline values for the selected lightcurve. The node
        # will automatically fill in the correct baseline value based on the index given by
        # the selected_lightcurve parameter.
        for fltr in self.all_filters:
            baselines = [lc.baseline.get(fltr, 0.0) for lc in lightcurves]
            baseline_selector = GivenValueSelector(baselines, self.selected_lightcurve)
            self.add_parameter(f"baseline_{fltr}", value=baseline_selector, allow_gradient=False)

    def __len__(self):
        """Get the number of lightcurves."""
        return len(self.lightcurves)

    @classmethod
    def from_lclib_file(cls, lightcurves_file, passbands, lc_t0=0.0, filters=None, **kwargs):
        """Create a MultiLightcurveSource from a lightcurves file in LCLIB format.

        Parameters
        ----------
        lightcurves_file : str
            The path to the lightcurves file in LCLIB format.
        passbands : Passband or PassbandGroup
            The passband or passband group to use for defining the lightcurve.
        lc_t0 : float
            The reference epoch (t0) of the input light curve. The model will be shifted
            to the model's t0 when computing fluxes.  For periodic lightcurves, this either
            must be set to the first time of the lightcurve or left as 0.0 to automatically
            derive the lc_t0 from the lightcurve.
            Default: 0.0
        filters : list of str, optional
            A list of filters to use for the lightcurves. If None, all filters will be used.
            Used to select a subset of filters that match the survey to simulate.
            Default: None
        **kwargs
            Additional keyword arguments to pass to the LightcurveData constructor, including
            the parameters for the model such as `dec`, `ra`, and `t0` and metadata
            such as `node_label`.

        Returns
        -------
        MultiLightcurveSource
            An instance of MultiLightcurveSource with the loaded lightcurves.
        """
        lightcurve_tables = read_lclib_data(lightcurves_file)
        if lightcurve_tables is None or len(lightcurve_tables) == 0:
            raise ValueError(f"Could not read lightcurves from file: {lightcurves_file}")

        lightcurves = []
        for table in lightcurve_tables:
            lc_data = LightcurveData.from_lclib_table(table, lc_t0=lc_t0, filters=filters)
            lightcurves.append(lc_data)

        return cls(lightcurves, passbands, **kwargs)

    def compute_flux(self, times, wavelengths, graph_state):
        """Draw effect-free observer frame flux densities.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray, optional
            A length N array of observer frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of observer frame SED values (in nJy). These are generated
            from non-overlapping box-shaped SED basis functions for each filter and
            scaled by the lightcurve values.
        """
        # Use the lightcurve selected by the sampler node to compute the flux density.
        model_ind = self.get_param(graph_state, "selected_lightcurve")
        return self.compute_flux_given_lc(
            self.lightcurves[model_ind],
            times,
            wavelengths,
            graph_state,
        )

    def compute_bandflux(self, times, filters, state, **kwargs):
        """Evaluate the model at the passband level for a single, given graph state.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        filters : numpy.ndarray
            A length T array of filter names.
        state : GraphState
            An object mapping graph parameters to their values with num_samples=1.
        **kwargs : dict
            Additional keyword arguments, not used in this method.

        Returns
        -------
        band_fluxes : numpy.ndarray
            A length T matrix of observer frame passband fluxes (in nJy).
        """
        params = self.get_local_params(state)
        model_ind = params["selected_lightcurve"]
        lc = self.lightcurves[model_ind]

        # Check that the filters are all supported by the model.
        for flt in np.unique(filters):
            if flt not in lc.lightcurves:
                raise ValueError(f"Filter '{flt}' is not supported by LightcurveSource {model_ind}.")

        # Shift the times for the model's t0 aligned with the lightcurve's lc_t0.
        # The lightcurve times were already shifted in the constructor to be relative to lc_t0.
        shifted_times = times - params["t0"]

        band_fluxes = np.zeros(len(times))
        for filter in lc.filters:
            filter_mask = filters == filter
            band_fluxes[filter_mask] = lc.evaluate(shifted_times[filter_mask], filter)

        return band_fluxes
