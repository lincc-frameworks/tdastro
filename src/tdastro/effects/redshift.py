from tdastro.base_models import EffectModel


class Redshift(EffectModel):
    """A redshift effect model.

    This contains a "pre-effect" method, which is used to calculate the emitted wavelengths/times
    needed to give us the observed wavelengths and times given the redshift.

    Attributes
    ----------
    pz : `float`
        The redshift.

    Notes
    -----
    Conversions used are as follows:
    - emitted_wavelength = observed_wavelength / (1 + redshift)
    - emitted_time = observed_time / (1 + redshift)
    - observed_flux = emitted_flux / (1 + redshift)
    """

    def __init__(self, pz, **kwargs):
        """Create a Redshift effect model.

        Parameters
        ----------
        pz : `float`
            The redshift.
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.pz = pz

    def pre_effect(self, observed_times, observed_wavelengths, **kwargs):
        """Calculate the emitted wavelengths/times needed to give us the observed wavelengths
        and times given the redshift.

        Parameters
        ----------
        observed_times : float
            The times at which the observation is made.
        observed_wavelengths : float
            The wavelengths at which the observation is made.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        float
            The adjusted flux density at the observed wavelength.
        """
        return (observed_times / (1 + self.pz), observed_wavelengths / (1 + self.pz))

    def apply(self, flux_density, bands=None, physical_model=None, **kwargs):
        """Apply the effect to observations (flux_density values).

        Parameters
        ----------
        flux_density : `numpy.ndarray`
            An array of flux density values.
        bands : `numpy.ndarray`, optional
            An array of bands.
        physical_model : `PhysicalModel`
            A PhysicalModel from which the effect may query parameters
            such as redshift, position, or distance.
        **kwargs : `dict`, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            The results.
        """
        return flux_density / (1 + self.pz)
