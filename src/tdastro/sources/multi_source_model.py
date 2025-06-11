"""Multiple source models wrap multiple PhysicalModels, allowing the
user to define such operations as additive models, where each source
contributes to the total flux density, or random source models, where
only one source is selected at random for each flux calculation.
"""

import numpy as np

from tdastro.graph_state import GraphState
from tdastro.math_nodes.given_sampler import GivenValueSampler
from tdastro.sources.physical_model import PhysicalModel


class MultiSourceModel(PhysicalModel):
    """A MultiSourceModel wraps multiple PhysicalModels.

    All rest frame effects are applied to each source, allowing different redshifts
    for each source (for unresolved sources).  The observer frame effects are applied
    to the weighted sum of the sources.

    Note: Each source may have its own sampled (RA, dec) position, which are not
    required to align.

    Attributes
    ----------
    sources : list
        A list of PhysicalModel objects to use in the flux calculation.
    num_sources : int
        The number of sources in the model.

    Parameters
    ----------
    sources : list
        A list of PhysicalModel objects to use in the flux calculation.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        sources,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Check that all sources are PhysicalModel objects and they do not contain
        # observer frame effects.
        for source in sources:
            if not isinstance(source, PhysicalModel):
                raise ValueError("All sources must be PhysicalModel objects.")
            if len(source.obs_frame_effects) > 0:
                raise ValueError("A MultiSourceModel cannot contain sources with observer frame effects.")
        self.sources = sources
        self.num_sources = len(sources)

    def set_graph_positions(self, seen_nodes=None):
        """Force an update of the graph structure (numbering of each node).

        Parameters
        ----------
        seen_nodes : set, optional
            A set of nodes that have already been processed to prevent infinite loops.
            Caller should not set.
        """
        if seen_nodes is None:
            seen_nodes = set()

        # Set the graph positions for this node and each source.
        super().set_graph_positions(seen_nodes=seen_nodes)
        for source in self.sources:
            source.set_graph_positions(seen_nodes=seen_nodes)

    def set_apply_redshift(self, apply_redshift):
        """Toggles the apply_redshift setting.

        Parameters
        ----------
        apply_redshift : bool
            The new value for apply_redshift.
        """
        for source in self.sources:
            source.set_apply_redshift(apply_redshift)

    def add_effect(self, effect):
        """Add an effect to the model.  Rest frame effects are applied to each source
        and observer frame effects are stored in this model.

        Parameters
        ----------
        effect : Effect
            The effect to add to the model.
        """
        if effect.rest_frame:
            # The rest frame effects are applied to each source.
            for source in self.sources:
                source.add_effect(effect)
        else:
            # Add observer frame effects to this model.
            # Add any effect parameters that are not already in the model.
            for param_name, setter in effect.parameters.items():
                if param_name not in self.setters:
                    self.add_parameter(param_name, setter, allow_gradient=False)
            self.obs_frame_effects.append(effect)

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

        # We use the same seen_nodes for all sampling calls so each node
        # is sampled at most one time regardless of link structure.
        graph_state = GraphState(num_samples)
        if given_args is not None:
            graph_state.update(given_args, all_fixed=True)

        seen_nodes = {}
        for source in self.sources:
            source._sample_helper(graph_state, seen_nodes, rng_info=rng_info)
        self._sample_helper(graph_state, seen_nodes, rng_info=rng_info)

        return graph_state

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
        raise NotImplementedError


class AdditiveMultiSourceModel(MultiSourceModel):
    """An AdditiveMultiSourceModel computes the flux from multiple overlapping objects,
    including (host, source pairs) or unresolved sources.

    All rest frame effects are applied to each source, allowing different redshifts
    for each source (for unresolved sources).  The observer frame effects are applied
    to the weighted sum of the sources.

    Note: Each source may have its own sampled (RA, dec) position, which are not
    required to align.

    Attributes
    ----------
    sources : list
        A list of PhysicalModel objects to use in the flux calculation.
    weights : numpy.ndarray, optional
        A length N array of weights to apply to each source. If None, all sources
        will be weighted equally.
    num_sources : int
        The number of sources in the model.

    Parameters
    ----------
    sources : list
        A list of PhysicalModel objects to use in the flux calculation.
    weights : numpy.ndarray, optional
        A length N array of weights to apply to each source. If None, all sources
        will be weighted equally.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        sources,
        weights=None,
        **kwargs,
    ):
        super().__init__(sources, **kwargs)

        if weights is None:
            self.weights = np.full(len(sources), 1.0)
        elif len(weights) != len(sources):
            raise ValueError("Length of weights must match length of sources.")
        else:
            self.weights = weights

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
        # Compute the weighted sum of contributions from each source.
        flux_density = np.zeros((len(times), len(wavelengths)))
        for source, weight in zip(self.sources, self.weights, strict=False):
            flux_density += weight * source._evaluate_single(
                times,
                wavelengths,
                state,
                rng_info=rng_info,
                **kwargs,
            )

        # Apply the observer frame effects on the weighted sum of sources.
        params = self.get_local_params(state)
        for effect in self.obs_frame_effects:
            flux_density = effect.apply(
                flux_density,
                times=times,
                wavelengths=wavelengths,
                rng_info=rng_info,
                **params,
            )

        return flux_density


class RandomMultiSourceModel(MultiSourceModel):
    """A RandomMultiSourceModel selects one of its sources at random and
    computes the flux from that source.

    Attributes
    ----------
    source_map : dict
        A dictionary mapping each source name (or index) to a PhysicalModel object.
    num_sources : int
        The number of sources in the model.

    Parameters
    ----------
    sources : list
        A list of PhysicalModel objects to use in the flux calculation.
    weights : numpy.ndarray, optional
        A length N array indicating the relative weight from which to select
        a source at random. If None, all sources will be weighted equally.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        sources,
        weights=None,
        source_names=None,
        **kwargs,
    ):
        super().__init__(sources, **kwargs)

        # Create a parameter to indicate which source was selected.
        source_names = source_names or [src.node_string for src in sources]
        self.source_map = {name: src for name, src in zip(source_names, sources, strict=False)}
        self._sampler_node = GivenValueSampler(source_names, weights=weights)
        self.add_parameter("selected_source", value=self._sampler_node, allow_gradient=False)

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
        # Use the model selected by the sampler node to compute the flux density.
        model_name = self.get_param(state, "selected_source")
        flux_density = self.source_map[model_name]._evaluate_single(
            times,
            wavelengths,
            state,
            rng_info=rng_info,
            **kwargs,
        )

        # Apply the observer frame effects on the weighted sum of sources.
        params = self.get_local_params(state)
        for effect in self.obs_frame_effects:
            flux_density = effect.apply(
                flux_density,
                times=times,
                wavelengths=wavelengths,
                rng_info=rng_info,
                **params,
            )

        return flux_density
