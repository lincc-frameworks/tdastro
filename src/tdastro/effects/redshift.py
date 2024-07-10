from tdastro.base_models import EffectModel


class Redshift(EffectModel):
    """A redshift effect model.

    This contains a "pre-effect" method, which is used to calculate the emitted wavelengths/times
    needed to give us the observed wavelengths and times given the redshift.

    Attributes
    ----------
    redshift : `float`
        The redshift.

    Notes
    -----
    Conversions used are as follows:
    - emitted_wavelength = observed_wavelength / (1 + redshift)
    - emitted_time = (t0 - observation_time) / (1 + redshift) + t0
    - observed_flux = emitted_flux / (1 + redshift)
    """

    def __init__(self, redshift=None, t0=None, **kwargs):
        """Create a Redshift effect model.

        Parameters
        ----------
        redshift : `float`
            The redshift.
        t0 : `float`
            The epoch of the peak or the zero phase, date. # TODO WORDING (1/?)
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_parameter("redshift", redshift, required=True, **kwargs)
        self.add_parameter("t0", t0, required=True, **kwargs)

    def required_parameters(self):  # TODO - can this just be an attribute?
        """Return the required parameters for the Redshift effect model."""
        return ["redshift", "t0"]

    def __str__(self) -> str:
        """Return a string representation of the Redshift effect model."""
        return f"RedshiftEffect(redshift={self.redshift})"

    def pre_effect(
        self, observed_times, observed_wavelengths, **kwargs
    ):  # TODO WORDING (2/?) -> should I change "emitted" to "rest"?
        """Calculate the emitted times and wavelengths needed to give us the observed times and wavelengths
        given the redshift.

        Parameters
        ----------
        observed_times : numpy.ndarray
            The times at which the observation is made.
        observed_wavelengths : numpy.ndarray
            The wavelengths at which the observation is made.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            The emission-frame times and wavelengths needed to generate the emission-frame flux densities,
            which will then be redshifted to observation-frame flux densities at the observation-frame
            times and wavelengths.
        """
        observed_times_rel_to_t0 = observed_times - self.t0
        emitted_times_rel_to_t0 = observed_times_rel_to_t0 / (1 + self.redshift)
        emitted_times = emitted_times_rel_to_t0 + self.t0
        emitted_wavelengths = observed_wavelengths / (1 + self.redshift)
        return (emitted_times, emitted_wavelengths)

    def apply(self, flux_density, wavelengths=None, physical_model=None, **kwargs):
        """Apply the effect to observations (flux_density values).

        Parameters
        ----------
        flux_density : `numpy.ndarray`
            A length T X N matrix of flux density values.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        physical_model : `PhysicalModel`
            A PhysicalModel from which the effect may query parameters such as redshift, position, or
            distance.
        **kwargs : `dict`, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            The results.
        """
        if physical_model is None:
            raise ValueError("No physical model provided to Redshift effect.")
        return flux_density / (1 + self.redshift)
