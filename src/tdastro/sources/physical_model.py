"""The base PhysicalModel used for all sources."""

import numpy as np

from tdastro.astro_utils.cosmology import RedshiftDistFunc
from tdastro.base_models import ParameterizedNode
from tdastro.graph_state import GraphState


class PhysicalModel(ParameterizedNode):
    """A physical model of a source of flux.

    Physical models can have fixed attributes (where you need to create a new model or use
    a setter function to change them) and settable model parameters that can be passed functions
    or constants and are stored in the graph's (external) graph_state dictionary.

    Physical models can also have special background pointers that link to another PhysicalModel
    producing flux. We can chain these to have a supernova in front of a star in front
    of a static background.

    Attributes
    ----------
    background : `PhysicalModel`
        A source of background flux such as a host galaxy.
    effects : `list`
        A list of effects to apply to an observations.

    Parameters
    ----------
    ra : `float`
        The object's right ascension (in degrees)
    dec : `float`
        The object's declination (in degrees)
    redshift : `float`
        The object's redshift.
    distance : `float`
        The object's luminosity distance (in pc). If no value is provided and
        a ``cosmology`` parameter is given, the model will try to derive from
        the redshift and the cosmology.
    background : `PhysicalModel`
        A source of background flux such as a host galaxy.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, ra=None, dec=None, redshift=None, distance=None, background=None, **kwargs):
        super().__init__(**kwargs)
        self.effects = []

        # Set RA, dec, and redshift from the parameters.
        self.add_parameter("ra", ra)
        self.add_parameter("dec", dec)
        self.add_parameter("redshift", redshift)

        # If the luminosity distance is provided, use that. Otherwise try the
        # redshift value using the cosmology (if given). Finally, default to None.
        if distance is not None:
            self.add_parameter("distance", distance)
        elif redshift is not None and kwargs.get("cosmology", None) is not None:
            self._redshift_func = RedshiftDistFunc(redshift=self.redshift, **kwargs)
            self.add_parameter("distance", self._redshift_func)
        else:
            self.add_parameter("distance", None)

        # Background is an object not a sampled parameter
        self.background = background

    def add_effect(self, effect, allow_dups=True, **kwargs):
        """Add a transformational effect to the PhysicalModel.
        Effects are applied in the order in which they are added.

        Parameters
        ----------
        effect : `EffectModel`
            The effect to apply.
        allow_dups : `bool`
            Allow multiple effects of the same type.
            Default = ``True``
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Raises
        ------
        Raises a ``AttributeError`` if the PhysicalModel does not have all of the
        required model parameters.
        """
        # Check that we have not added this effect before.
        if not allow_dups:
            effect_type = type(effect)
            for prev in self.effects:
                if effect_type == type(prev):
                    raise ValueError("Added the effect type to a model {effect_type} more than once.")

        self.effects.append(effect)

        # Reset the node position to indicate the graph has changed.
        self.node_pos = None

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
        for effect in self.effects:
            effect.set_graph_positions(seen_nodes=seen_nodes)

    def _evaluate(self, times, wavelengths, graph_state):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of rest frame timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        graph_state : `GraphState`
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values.
        """
        raise NotImplementedError()

    def evaluate(self, times, wavelengths, graph_state=None, given_args=None, rng_info=None, **kwargs):
        """Draw observations for this object and apply the noise.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of observer frame timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        graph_state : `GraphState`, optional
            An object mapping graph parameters to their values.
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        rng_info : `dict`, optional
            A dictionary of random number generator information for each node, such as
            the JAX keys or the numpy rngs.
        **kwargs : `dict`, optional
            All the other keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values.
        """
        # Make sure times and wavelengths are numpy arrays.
        times = np.array(times)
        wavelengths = np.array(wavelengths)

        # Check if we need to sample the graph.
        if graph_state is None:
            graph_state = self.sample_parameters(
                given_args=given_args, num_samples=1, rng_info=rng_info, **kwargs
            )
        params = self.get_local_params(graph_state)

        # Pre-effects are adjustments done to times and/or wavelengths, before flux density computation.
        for effect in self.effects:
            if hasattr(effect, "pre_effect"):
                times, wavelengths = effect.pre_effect(times, wavelengths, graph_state, **kwargs)

        # Compute the flux density for both the current object and add in anything
        # behind it, such as a host galaxy.
        flux_density = self._evaluate(times, wavelengths, graph_state, **kwargs)
        if self.background is not None:
            flux_density += self.background._evaluate(
                times,
                wavelengths,
                graph_state,
                ra=params["ra"],
                dec=params["dec"],
                **kwargs,
            )

        for effect in self.effects:
            flux_density = effect.apply(flux_density, wavelengths, graph_state, **kwargs)
        return flux_density

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
        rng_info : `dict`, optional
            A dictionary of random number generator information for each node, such as
            the JAX keys or the numpy rngs.
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

        args_to_use = {}
        if given_args is not None:
            args_to_use.update(given_args)
        if kwargs is not None:
            args_to_use.update(kwargs)

        # We use the same seen_nodes for all sampling calls so each node
        # is sampled at most one time regardless of link structure.
        graph_state = GraphState(num_samples)
        seen_nodes = {}
        if self.background is not None:
            self.background._sample_helper(graph_state, seen_nodes, args_to_use, rng_info, **kwargs)
        self._sample_helper(graph_state, seen_nodes, args_to_use, rng_info, **kwargs)

        for effect in self.effects:
            effect._sample_helper(graph_state, seen_nodes, args_to_use, rng_info, **kwargs)

        return graph_state

    def get_all_node_info(self, field, seen_nodes=None):
        """Return a list of requested information for each node.

        Parameters
        ----------
        field : `str`
            The name of the attribute to extract from the node.
            Common examples are: "node_hash" and "node_string"
        seen_nodes : `set`
            A set of objects that have already been processed.
            Modified in place if provided.

        Returns
        -------
        result : `list`
            A list of values for each unique node in the graph.
        """
        # Check if we have already processed this node.
        if seen_nodes is None:
            seen_nodes = set()

        # Get the information for this node, the background, all effects,
        # and each of their dependencies.
        result = super().get_all_node_info(field, seen_nodes)
        if self.background is not None:
            result.extend(self.background.get_all_node_info(field, seen_nodes))
        for effect in self.effects:
            result.extend(effect.get_all_node_info(field, seen_nodes))
        return result
