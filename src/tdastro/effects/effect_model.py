"""The base EffectModel class used for all effects."""


class EffectModel:
    """A physical or systematic effect to apply to an observation.

    Effects are not ParameterizedNodes but can have arguments that are
    ParameterizedNodes. These arguments are stored as parameters in the
    source node (PhysicalModel) when add_effect() is called. This allows
    the effect to easily use existing parameters from the source node as
    well as new parameters specific to the effect. The source node samples
    these parameters and passes them in the call to apply().

    The EffectModel class is designed to be subclassed, and the
    subclasses should implement the apply() method to define the
    specific effect being applied.

    Attributes
    ----------
    rest_frame : bool
        Whether the effect is applied in the rest frame of the observation (True)
        or in the observed frame (False).
    parameters : dict
        A dictionary of parameters for the effect. Maps the parameter names to
        their setters.
    """

    def __init__(self, rest_frame=True, **kwargs):
        self.rest_frame = rest_frame

        # Automatically include all keyword arguments as settable parameters.
        self.parameters = {}
        for key, value in kwargs.items():
            self.add_effect_parameter(key, value)

    def add_effect_parameter(self, name, setter):
        """Add a parameter to the effect.

        Note
        ----
        These parameters are automatically added to the corresponding source
        nodes so they are sampled and recorded with the model's other parameters.

        Parameters
        ----------
        name : str
            The name of the parameter.
        setter : function
            A function that sets the parameter value.
        """
        self.parameters[name] = setter

    def apply(self, flux_density, times=None, wavelengths=None, rng_info=None, **kwargs):
        """Apply the effect to observations (flux_density values).

        Parameters
        ----------
        flux_density : numpy.ndarray
            A length T X N matrix of flux density values (in nJy).
        times : numpy.ndarray, optional
            A length T array of times (in MJD).
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of flux densities after the effect is applied (in nJy).
        """
        raise NotImplementedError()

    def apply_bandflux(self, bandfluxes, *, times=None, filters=None, rng_info=None, **kwargs):
        """Apply the effect to band fluxes.

        Parameters
        ----------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes (in nJy).
        times : numpy.ndarray, optional
            A length T array of times (in MJD).
        filters : numpy.ndarray, optional
            A length N array of filters. If not provided, the effect is applied to all
            band fluxes.
        rng_info : numpy.random._generator.Generator, optional
            A given numpy random number generator to use for this computation. If not
            provided, the function uses the node's random number generator.
        **kwargs : `dict`, optional
           Any additional keyword arguments, including any additional
           parameters needed to apply the effect.

        Returns
        -------
        bandfluxes : numpy.ndarray
            A length T array of band fluxes after the effect is applied (in nJy).
        """
        raise NotImplementedError()
