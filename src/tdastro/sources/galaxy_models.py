import numpy as np
from astropy.coordinates import angular_separation

from tdastro.sources.physical_model import PhysicalModel


class GaussianGalaxy(PhysicalModel):
    """A static source.

    Attributes
    ----------
    radius_std : `float`
        The standard deviation of the brightness as we move away
        from the galaxy's center (in degrees).
    brightness : `float`
        The inherent brightness at the center of the galaxy.
    """

    def __init__(self, brightness, radius, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("galaxy_radius_std", radius, required=True, **kwargs)
        self.add_parameter("brightness", brightness, required=True, **kwargs)

    def sample_ra(self):
        """Sample an right ascension coordinate based on the center and radius of the galaxy.

        Returns
        -------
        ra : `float`
            The sampled right ascension in degrees.
        """
        return np.random.normal(loc=self.ra, scale=self.galaxy_radius_std)

    def sample_dec(self):
        """Sample a declination coordinate based on the center and radius of the galaxy.

        Returns
        -------
        dec : `float`
            The sampled declination in degrees.
        """
        return np.random.normal(loc=self.dec, scale=self.galaxy_radius_std)

    def _evaluate(self, times, wavelengths, ra=None, dec=None, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        ra : `float`, optional
            The right ascension of the observations in degrees.
        dec : `float`, optional
            The declination of the observations in degrees.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values.
        """
        if ra is None:
            ra = self.ra
        if dec is None:
            dec = self.dec

        # Scale the brightness as a Guassian function centered on the object's RA and Dec.
        dist = angular_separation(
            self.ra * np.pi / 180.0,
            self.dec * np.pi / 180.0,
            ra * np.pi / 180.0,
            dec * np.pi / 180.0,
        )
        scale = np.exp(-(dist * dist) / (2.0 * self.galaxy_radius_std * self.galaxy_radius_std))

        return np.full((len(times), len(wavelengths)), self.brightness * scale)
