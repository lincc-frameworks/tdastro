import numpy as np
from astropy.coordinates import angular_separation

from tdastro.sources.physical_model import PhysicalModel


class GaussianGalaxy(PhysicalModel):
    """A static source.

    Parameters
    ----------
    radius_std : `float`
        The standard deviation of the brightness as we move away
        from the galaxy's center (in degrees).
    brightness : `float`
        The inherent brightness at the center of the galaxy.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, brightness, radius, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("galaxy_radius_std", radius, required=True, **kwargs)
        self.add_parameter("brightness", brightness, required=True, **kwargs)

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
        dist = 0.0
        if ra is not None and dec is not None:
            dist = angular_separation(
                self.parameters["ra"] * np.pi / 180.0,
                self.parameters["dec"] * np.pi / 180.0,
                ra * np.pi / 180.0,
                dec * np.pi / 180.0,
            )

        # Scale the brightness as a Guassian function centered on the object's RA and Dec.
        std = self.parameters["galaxy_radius_std"]
        scale = np.exp(-(dist * dist) / (2.0 * std * std))

        return np.full((len(times), len(wavelengths)), self.parameters["brightness"] * scale)
