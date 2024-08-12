"""The base PhysicalModel used for all sources."""

from tdastro.astro_utils.cosmology import RedshiftDistFunc
from tdastro.base_models import ParameterizedNode


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

    def _evaluate(self, times, wavelengths, graph_state):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        graph_state : `dict`
            A dictionary mapping graph parameters to their values.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values.
        """
        raise NotImplementedError()

    def evaluate(self, times, wavelengths, graph_state=None, **kwargs):
        """Draw observations for this object and apply the noise.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        graph_state : `dict`, optional
            A given setting of all the parameters and their values. If this is not
            included then a random sampling is used.
        **kwargs : `dict`, optional
            All the other keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values.
        """
        if graph_state is None:
            graph_state = self.sample_parameters()
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

    def sample_parameters(self, given_args=None, **kwargs):
        """Sample the model's underlying parameters if they are provided by a function
        or ParameterizedModel.

        Parameters
        ----------
        given_args : `dict`, optional
            A dictionary representing the given arguments for this sample run.
            This can be used as the JAX PyTree for differentiation.
        **kwargs : `dict`, optional
            All the keyword arguments, including the values needed to sample
            parameters.

        Returns
        -------
        graph_state : `dict`
            A dictionary mapping graph parameters to their values.
        """
        # If the graph has not been sampled ever, update the node positions for
        # every node (model, background, effects).
        if self._node_pos is None:
            nodes = set()
            self.update_graph_information(seen_nodes=nodes)
            if self.background is not None:
                self.background.update_graph_information(seen_nodes=nodes)
            for effect in self.effects:
                effect.update_graph_information(seen_nodes=nodes)

        args_to_use = {}
        if given_args is not None:
            args_to_use.update(given_args)
        if kwargs is not None:
            args_to_use.update(kwargs)

        # We use the same seen_nodes for all sampling calls so each node
        # is sampled at most one time regardless of link structure.
        graph_state = {}
        seen_nodes = {}
        if self.background is not None:
            self.background._sample_helper(graph_state, seen_nodes, args_to_use, **kwargs)
        self._sample_helper(graph_state, seen_nodes, args_to_use, **kwargs)

        for effect in self.effects:
            effect._sample_helper(graph_state, seen_nodes, args_to_use, **kwargs)

        return graph_state
