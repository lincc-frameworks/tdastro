"""The base PhysicalModel used for all sources."""

import numpy as np

from tdastro.astro_utils.passbands import Passband
from tdastro.astro_utils.redshift import RedshiftDistFunc, obs_to_rest_times_waves, rest_to_obs_flux
from tdastro.base_models import ParameterizedNode
from tdastro.graph_state import GraphState


class PhysicalModel(ParameterizedNode):
    """A physical model of a source of flux.

    Physical models can have fixed attributes (where you need to create a new model or use
    a setter function to change them) and settable model parameters that can be passed functions
    or constants and are stored in the graph's (external) graph_state dictionary.

    Physical models can also have special background pointers that link to another PhysicalModel
    producing flux density. We can chain these to have a supernova in front of a star in front
    of a static background.

    Physical models also support adding and applying a variety of effects, such as redshift.

    Parameterized values include:
      * dec - The object's declination in degrees.
      * distance - The object's luminosity distance in pc.
      * ra - The object's right ascension in degrees.
      * redshift - The object's redshift.
      * t0 - The t0 of the zero phase (if applicable), date.

    Attributes
    ----------
    background : `PhysicalModel`
        A source of background flux such as a host galaxy.
    apply_redshift : `bool`
        Indicates whether to apply the redshift.

    Parameters
    ----------
    ra : `float`
        The object's right ascension (in degrees)
    dec : `float`
        The object's declination (in degrees)
    redshift : `float`
        The object's redshift.
    t0 : `float`
        The phase offset in MJD. For non-time-varying phenomena, this has no effect.
    distance : `float`
        The object's luminosity distance (in pc). If no value is provided and
        a ``cosmology`` parameter is given, the model will try to derive from
        the redshift and the cosmology.
    background : `PhysicalModel`
        A source of background flux such as a host galaxy.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, ra=None, dec=None, redshift=None, t0=None, distance=None, background=None, **kwargs):
        super().__init__(**kwargs)

        # Set RA, dec, and redshift from the parameters.
        self.add_parameter("ra", ra, allow_gradient=False)
        self.add_parameter("dec", dec, allow_gradient=False)
        self.add_parameter("redshift", redshift, allow_gradient=False)
        self.add_parameter("t0", t0)

        # If the luminosity distance is provided, use that. Otherwise try the
        # redshift value using the cosmology (if given). Finally, default to None.
        if distance is not None:
            self.add_parameter("distance", distance, allow_gradient=False)
        elif redshift is not None and kwargs.get("cosmology", None) is not None:
            self._redshift_func = RedshiftDistFunc(redshift=self.redshift, **kwargs)
            self.add_parameter("distance", self._redshift_func, allow_gradient=False)
        else:
            self.add_parameter("distance", None, allow_gradient=False)

        # Background is an object not a sampled parameter
        self.background = background

        # Initialize the effect settings to their default values.
        self.apply_redshift = redshift is not None

    def set_graph_positions(self, seen_nodes=None):
        """Force an update of the graph structure (numbering of each node).

        Parameters
        ----------
        seen_nodes : `set`, optional
            A set of nodes that have already been processed to prevent infinite loops.
            Caller should not set.
        """
        if seen_nodes is None:
            seen_nodes = set()

        # Set the graph positions for each node, its background, and all of its effects.
        super().set_graph_positions(seen_nodes=seen_nodes)
        if self.background is not None:
            self.background.set_graph_positions(seen_nodes=seen_nodes)

    def set_apply_redshift(self, apply_redshift):
        """Toggles the apply_redshift setting.

        Parameters
        ----------
        apply_redshift : `bool`
            The new value for apply_redshift.
        """
        self.apply_redshift = apply_redshift

    def mask_by_time(self, times, graph_state=None):
        """Compute a mask for whether a given time is of interest for a given object.
        For example, a user can use this function to generate a mask to include
        only the observations of interest for a window around the supernova.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        graph_state : GraphState, optional
            An object mapping graph parameters to their values.

        Returns
        -------
        time_mask : numpy.ndarray
            A length T array of Booleans indicating whether the time is of interest.
        """
        return np.full(len(times), True)

    def compute_flux(self, times, wavelengths, graph_state):
        """Draw effect-free rest frame flux densities.
        The rest-frame flux is defined as F_nu = L_nu / 4*pi*D_L**2,
        where D_L is the luminosity distance.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of rest frame timestamps in MJD.
        wavelengths : `numpy.ndarray`, optional
            A length N array of rest frame wavelengths (in angstroms).
        graph_state : `GraphState`
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of rest frame SED values (in nJy).
        """
        raise NotImplementedError()

    def evaluate(self, times, wavelengths, graph_state=None, given_args=None, rng_info=None, **kwargs):
        """Draw observations for this object and apply the noise.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of observer frame timestamps in MJD.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths (in angstroms).
        graph_state : `GraphState`, optional
            An object mapping graph parameters to their values.
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : `dict`, optional
            All the other keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length S x T x N matrix of SED values (in nJy), where S is the number of samples,
            T is the number of time steps, and N is the number of wavelengths.
            If S=1 then the function returns a T x N matrix.
        """
        # Make sure times and wavelengths are numpy arrays.
        times = np.asarray(times)
        wavelengths = np.asarray(wavelengths)

        # Check if we need to sample the graph.
        if graph_state is None:
            graph_state = self.sample_parameters(
                given_args=given_args, num_samples=1, rng_info=rng_info, **kwargs
            )

        results = np.empty((graph_state.num_samples, len(times), len(wavelengths)))
        for sample_num, state in enumerate(graph_state):
            params = self.get_local_params(state)

            # Pre-effects are adjustments done to times and/or wavelengths, before flux density
            # computation. We skip if redshift is 0.0 since there is nothing to do.
            if self.apply_redshift and params["redshift"] != 0.0:
                if params.get("redshift", None) is None:
                    raise ValueError("The 'redshift' parameter is required for redshifted models.")
                if params.get("t0", None) is None:
                    raise ValueError("The 't0' parameter is required for redshifted models.")
                times, wavelengths = obs_to_rest_times_waves(
                    times, wavelengths, params["redshift"], params["t0"]
                )

            # Compute the flux density for both the current object and add in anything
            # behind it, such as a host galaxy.
            flux_density = self.compute_flux(times, wavelengths, state, **kwargs)
            if self.background is not None:
                flux_density += self.background.compute_flux(
                    times,
                    wavelengths,
                    state,
                    ra=params["ra"],
                    dec=params["dec"],
                    **kwargs,
                )

            # Post-effects are adjustments done to the flux density after computation.
            if self.apply_redshift and params["redshift"] != 0.0:
                # We have alread checked that redshift is not None.
                flux_density = rest_to_obs_flux(flux_density, params["redshift"])

            # Save the result.
            results[sample_num, :, :] = flux_density

        if graph_state.num_samples == 1:
            return results[0, :, :]
        return results

    def sample_parameters(self, given_args=None, num_samples=1, rng_info=None, **kwargs):
        """Sample the model's underlying parameters if they are provided by a function
        or ParameterizedModel.

        Parameters
        ----------
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        num_samples : `int`
            A count of the number of samples to compute.
            Default: 1
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : `dict`, optional
            All the keyword arguments, including the values needed to sample
            parameters.

        Returns
        -------
        graph_state : `GraphState`
            An object mapping graph parameters to their values.
        """
        # If the graph has not been sampled ever, update the node positions for
        # every node (model, background, effects).
        if self.node_pos is None:
            self.set_graph_positions()

        # We use the same seen_nodes for all sampling calls so each node
        # is sampled at most one time regardless of link structure.
        graph_state = GraphState(num_samples)
        if given_args is not None:
            graph_state.update(given_args, all_fixed=True)

        seen_nodes = {}
        if self.background is not None:
            self.background._sample_helper(graph_state, seen_nodes, rng_info, **kwargs)
        self._sample_helper(graph_state, seen_nodes, rng_info, **kwargs)

        return graph_state

    def get_band_fluxes(self, passband_or_group, times, filters, state) -> np.ndarray:
        """Get the band fluxes for a given Passband or PassbandGroup.

        Parameters
        ----------
        passband_or_group : `Passband` or `PassbandGroup`
            The passband (or passband group) to use.
        times : `numpy.ndarray`
            A length T array of observer frame timestamps in MJD.
        filters : `numpy.ndarray` or None
            A length T array of filter names. It may be None if
            passband_or_group is a Passband.
        state : `GraphState`
            An object mapping graph parameters to their values.

        Returns
        -------
        band_fluxes : `numpy.ndarray`
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
            spectral_fluxes = self.evaluate(times, passband_or_group.waves, state)
            return passband_or_group.fluxes_to_bandflux(spectral_fluxes)

        if filters is None:
            raise ValueError("If passband_or_group is a PassbandGroup, filters must be provided.")

        band_fluxes = np.empty((state.num_samples, len(times)))
        for filter_name in np.unique(filters):
            passband = passband_or_group[filter_name]
            filter_mask = filters == filter_name
            spectral_fluxes = self.evaluate(times[filter_mask], passband.waves, state)
            band_fluxes[:, filter_mask] = passband.fluxes_to_bandflux(spectral_fluxes)

        if state.num_samples == 1:
            return band_fluxes[0, :]
        return band_fluxes
