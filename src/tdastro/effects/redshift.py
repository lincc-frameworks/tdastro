from tdastro.effects.effect_model import EffectModel


class Redshift(EffectModel):
    """A redshift effect model.

    This contains a "pre-effect" method, which is used to calculate the emitted wavelengths/times
    needed to give us the observed wavelengths and times given the redshift. Times are calculated
    with respect to the t0 of the given model.

    Notes
    -----
    Conversions used are as follows:
    - rest_frame_wavelength = observation_frame_wavelength / (1 + redshift)
    - rest_frame_time = (observation_frame_time - t0) / (1 + redshift) + t0
    - observation_frame_flux = rest_frame_flux / (1 + redshift)
    """

    def __init__(self, redshift=None, t0=None, **kwargs):
        """Create a Redshift effect model.

        Parameters
        ----------
        redshift : `float`
            The redshift.
        t0 : `float`
            The reference epoch; e.g. the time of the maximum light of a supernova or the epoch of zero phase
            for a periodic variable star.
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_parameter("redshift", redshift, required=True, **kwargs)
        self.add_parameter("t0", t0, required=True, **kwargs)

    def pre_effect(self, observer_frame_times, observer_frame_wavelengths, **kwargs):
        """Calculate the rest-frame times and wavelengths needed to give us the observer-frame times
        and wavelengths (given the redshift).

        Parameters
        ----------
        observer_frame_times : numpy.ndarray
            The times at which the observation is made.
        observer_frame_wavelengths : numpy.ndarray
            The wavelengths at which the observation is made.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            The rest-frame times and wavelengths needed to generate the rest-frame flux densities,
            which will later be redshifted  back to observer-frame flux densities at the observer-frame
            times and wavelengths.
        """
        observed_times_rel_to_t0 = observer_frame_times - self.t0
        rest_frame_times_rel_to_t0 = observed_times_rel_to_t0 / (1 + self.redshift)
        rest_frame_times = rest_frame_times_rel_to_t0 + self.t0
        rest_frame_wavelengths = observer_frame_wavelengths / (1 + self.redshift)
        return (rest_frame_times, rest_frame_wavelengths)

    def apply(self, flux_density, wavelengths, physical_model=None, **kwargs):
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
            The redshifted results.
        """
        return flux_density / (1 + self.redshift)
