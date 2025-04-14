"""The base PhysicalModel used for all sources."""

from os import urandom

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

    Physical models also support adding and applying a variety of effects, such as redshift.

    Parameterized values include:
      * dec - The object's declination in degrees.
      * distance - The object's luminosity distance in pc.
      * ra - The object's right ascension in degrees.
      * redshift - The object's redshift.
      * t0 - The t0 of the zero phase (if applicable), date.

    Attributes
    ----------
    apply_redshift : bool
        Indicates whether to apply the redshift.
    rest_frame_effects : list of EffectModel
        A list of effects to apply in the rest frame.
    obs_frame_effects : list of EffectModel
        A list of effects to apply in the observer frame.

    Parameters
    ----------
    ra : float
        The object's right ascension (in degrees)
    dec : float
        The object's declination (in degrees)
    redshift : float
        The object's redshift.
    t0 : float
        The phase offset in MJD. For non-time-varying phenomena, this has no effect.
    distance : float
        The object's luminosity distance (in pc). If no value is provided and
        a cosmology parameter is given, the model will try to derive from
        the redshift and the cosmology.
    seed : int, optional
        The seed for a random number generator.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        ra=None,
        dec=None,
        redshift=None,
        t0=None,
        distance=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Set the parameters for the model.
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

        # Initialize the effect settings to their default values.
        self.apply_redshift = redshift is not None

        # Set the default effects.
        self.rest_frame_effects = []
        self.obs_frame_effects = []

        # Get a default random number generator for this object, using the
        # given seed if one is provided.
        if seed is None:
            seed = int.from_bytes(urandom(4), "big")
        self._rng = np.random.default_rng(seed=seed)

    def set_apply_redshift(self, apply_redshift):
        """Toggles the apply_redshift setting.

        Parameters
        ----------
        apply_redshift : bool
            The new value for apply_redshift.
        """
        self.apply_redshift = apply_redshift

    def add_effect(self, effect):
        """Add an effect to the model.

        Parameters
        ----------
        effect : EffectModel
            The effect to add.
        """
        # Add any effect parameters that are not already in the model.
        for param_name, setter in effect.parameters.items():
            if param_name not in self.setters:
                self.add_parameter(param_name, setter, allow_gradient=False)

        # Add the effect to the appropriate list.
        if effect.rest_frame:
            self.rest_frame_effects.append(effect)
        else:
            self.obs_frame_effects.append(effect)

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
        raise NotImplementedError()

    def _evaluate_single(self, times, wavelengths, state, rng_info=None, **kwargs):
        """Evaluate the model and apply the effects for a single, given graph state.
        This function applies redshift, computes the flux density for the object,
        applies rest frames effects, performs the redshift correction (if needed),
        and applies the observer frame effects.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray
            A length N array of wavelengths (in angstroms).
        state : GraphState
            An object mapping graph parameters to their values with num_samples=1.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            All the other keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        if state is None or state.num_samples != 1:
            raise ValueError("A GraphState with num_samples=1 required.")
        params = self.get_local_params(state)

        # Pre-effects are adjustments done to times and/or wavelengths, before flux density
        # computation. We skip if redshift is 0.0 since there is nothing to do.
        if self.apply_redshift and params["redshift"] != 0.0:
            if params.get("redshift", None) is None:
                raise ValueError("The 'redshift' parameter is required for redshifted models.")
            if params.get("t0", None) is None:
                raise ValueError("The 't0' parameter is required for redshifted models.")
            rest_times, rest_wavelengths = obs_to_rest_times_waves(
                times, wavelengths, params["redshift"], params["t0"]
            )
        else:
            rest_times = times
            rest_wavelengths = wavelengths

        # Compute the flux density for the object and apply any rest frame effects.
        flux_density = self.compute_flux(rest_times, rest_wavelengths, state, **kwargs)
        for effect in self.rest_frame_effects:
            flux_density = effect.apply(
                flux_density,
                times=rest_times,
                wavelengths=rest_wavelengths,
                rng_info=rng_info,
                **params,
            )

        # Post-effects are adjustments done to the flux density after computation.
        if self.apply_redshift and params["redshift"] != 0.0:
            # We have alread checked that redshift is not None.
            flux_density = rest_to_obs_flux(flux_density, params["redshift"])

        # Apply observer frame effects.
        for effect in self.obs_frame_effects:
            flux_density = effect.apply(
                flux_density,
                times=times,
                wavelengths=wavelengths,
                rng_info=rng_info,
                **params,
            )

        return flux_density

    def evaluate(self, times, wavelengths, graph_state=None, given_args=None, rng_info=None, **kwargs):
        """Draw observations for this object and apply the noise.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray
            A length N array of wavelengths (in angstroms).
        graph_state : GraphState, optional
            An object mapping graph parameters to their values.
        given_args : dict, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            All the other keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
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
            # Compute the flux (applying all effects) and save the result.
            results[sample_num, :, :] = self._evaluate_single(
                times,
                wavelengths,
                state,
                rng_info=rng_info,
                **kwargs,
            )

        if graph_state.num_samples == 1:
            return results[0, :, :]
        return results

    def sample_parameters(self, given_args=None, num_samples=1, rng_info=None):
        """Sample the model's underlying parameters if they are provided by a function
        or ParameterizedModel.

        Parameters
        ----------
        given_args : dict, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        num_samples : int
            A count of the number of samples to compute.
            Default: 1
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : dict, optional
            All the keyword arguments, including the values needed to sample
            parameters.

        Returns
        -------
        graph_state : GraphState
            An object mapping graph parameters to their values.
        """
        # If the graph has not been sampled ever, update the node positions for
        # every node (model, background, effects).
        if self.node_pos is None:
            self.set_graph_positions()

        graph_state = GraphState(num_samples)
        if given_args is not None:
            graph_state.update(given_args, all_fixed=True)

        # We use the same seen_nodes for all sampling calls so each node
        # is sampled at most one time regardless of link structure.
        seen_nodes = {}
        self._sample_helper(graph_state, seen_nodes, rng_info=rng_info)

        return graph_state

    def get_band_fluxes(self, passband_or_group, times, filters, state) -> np.ndarray:
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
            spectral_fluxes = self.evaluate(times, passband_or_group.waves, state)
            return passband_or_group.fluxes_to_bandflux(spectral_fluxes)

        if filters is None:
            raise ValueError("If passband_or_group is a PassbandGroup, filters must be provided.")
        filters = np.asarray(filters)

        band_fluxes = np.empty((state.num_samples, len(times)))
        for filter_name in np.unique(filters):
            passband = passband_or_group[filter_name]
            filter_mask = filters == filter_name
            spectral_fluxes = self.evaluate(times[filter_mask], passband.waves, state)
            band_fluxes[:, filter_mask] = passband.fluxes_to_bandflux(spectral_fluxes)

        if state.num_samples == 1:
            return band_fluxes[0, :]
        return band_fluxes
