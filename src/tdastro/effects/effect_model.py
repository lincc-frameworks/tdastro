"""The base EffectModel class used for all effects."""

import numpy as np


class EffectModel:
    """A physical or systematic effect to apply to an observation.

    Effects are not ParameterizedNodes, but rather get their arguments
    from the PhysicalObject's parameters via the kwargs in apply().

    Attributes
    ----------
    rest_frame : bool
        Whether the effect is applied in the rest frame of the observation (True)
        or in the observed frame (False).
    parameters : dict
        A dictionary of parameters for the effect. Mapps parameter names to
        their setters.
    """

    def __init__(self, rest_frame=True, **kwargs):
        self.rest_frame = rest_frame

        self.parameters = {}
        for key, value in kwargs.items():
            self.add_effect_parameter(key, value)

    def add_effect_parameter(self, name, setter):
        """Add a parameter to the effect.

        Parameters
        ----------
        name : str
            The name of the parameter.
        setter : function
            A function that sets the parameter value.
        """
        self.parameters[name] = setter

        # If we are just setting a scalar, set the attribute directly.
        if np.isscalar(setter):
            setattr(self, name, setter)

    def lookup_effect_parameter(self, name, **kwargs):
        """Look up a parameter value from either the keyword arguments
        or the effect's attributes.

        Parameters
        ----------
        name : str
            The name of the parameter.
        **kwargs : dict
            Additional keyword arguments to search for the parameter.

        Returns
        -------
        value : object
            The value of the parameter.
        """
        if name in kwargs:
            return kwargs[name]
        elif hasattr(self, name):
            return getattr(self, name)
        else:
            raise ValueError(f"{self.__class__.__name__} effect requires {name} parameter.")

    def apply(self, flux_density, rng_info=None, **kwargs):
        """Apply the effect to observations (flux_density values)

        Parameters
        ----------
        flux_density : numpy.ndarray
            A length T X N matrix of flux density values (in nJy).
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : `dict`, optional
           Any additional keyword arguments. This includes all of the
           parameters needed to apply the effect.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        raise NotImplementedError()
