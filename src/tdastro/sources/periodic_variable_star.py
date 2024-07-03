from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np
from astropy.modeling.physical_models import BlackBody

from tdastro.sources.periodic_source import PeriodicSource

PARSEC_TO_CM = (1 * u.pc).to_value(u.cm)


class PeriodicVariableStar(PeriodicSource, ABC):
    """A model for a periodic variable star.

    Attributes
    ----------
    period : `float`
        The period of the source, in days.
    epoch : `float`
        The epoch of the zero phase, date.
    distance : `float`
        The distance to the source, in pc.
    """

    def __init__(self, period, epoch, **kwargs):
        super().__init__(period, epoch, **kwargs)
        self.add_parameter("distance", required=True, **kwargs)

    def _evaluate_phases(self, phases, wavelengths, **kwargs):
        """Draw effect-free observations for this object, as a function of phase.

        Parameters
        ----------
        phases : `numpy.ndarray`
            A length T array of phases, in the range [0, 1].
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        **kwargs : `dict`, optional
              Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values.
        """
        distance = self.distance * PARSEC_TO_CM
        return self._luminosity_density_phases(phases, wavelengths, **kwargs) / (4 * np.pi * distance**2)

    @abstractmethod
    def _luminosity_density_phases(self, phases, wavelengths, **kwargs):
        """Draw effect-free luminosity density for this object, as a function of phase.

        Parameters
        ----------
        phases : `numpy.ndarray`
            A length T array of phases, in the range [0, 1].
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        **kwargs : `dict`, optional
              Any additional keyword arguments.

        Returns
        -------
        luminosity_density : `numpy.ndarray`
            A length T x N matrix of luminosity density values.
        """
        raise NotImplementedError()


class EclipsingBinaryStar(PeriodicVariableStar):
    """A toy model for a detached eclipsing binary star.

    It is assumed that the stars are spherical, SED is black-body,
    and the orbits are circular. Epoch is the time of the primary eclipse.
    No limb darkening, reflection, or other effects are included.

    Attributes
    ----------
    period : `float`
        The period of the source, in days.
    major_semiaxis : `float`
        The major semiaxis of the orbit, in AU.
    inclination : `float`
        The inclination of the orbit, in degrees.
    primary_radius : `float`
        The radius of the primary star, in solar radii.
    secondary_radius : `float`
        The radius of the secondary star, in solar radii.
    primary_temperature : `float`
        The effective temperature of the primary star, in kelvins.
    secondary_temperature : `float`
        The effective temperature of the secondary star, in kelvins.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter("major_semiaxis", required=True, **kwargs)
        self.add_parameter("inclination", required=True, **kwargs)
        self.add_parameter("primary_radius", required=True, **kwargs)
        self.add_parameter("secondary_radius", required=True, **kwargs)
        self.add_parameter("primary_temperature", required=True, **kwargs)
        self.add_parameter("secondary_temperature", required=True, **kwargs)

    def _luminosity_density_phases(self, phases, wavelengths, **kwargs):
        """Draw effect-free luminosity density for this object, as a function of phase.

        Parameters
        ----------
        phases : `numpy.ndarray`
            A length T array of phases, in the range [0, 1].
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths.
        **kwargs : `dict`, optional
              Any additional keyword arguments.

        Returns
        -------
        luminosity_density : `numpy.ndarray`
            A length T x N matrix of luminosity density values.
        """
        phases = phases[:, None]

        primary_lum_density = black_body_luminosity_density(
            self.primary_temperature, self.primary_radius, wavelengths
        )
        secondary_lum_density = black_body_luminosity_density(
            self.primary_temperature, self.primary_radius, wavelengths
        )

        # Distance between the centers of the stars on the plane of the sky
        star_centers_distance = self.major_semiaxis * self._norm_star_center_distance(
            phases, self.inclination
        )

        overlap_area = self._circle_overlap_area(
            star_centers_distance, self.primary_radius, self.secondary_radius
        )

        # For phases around the main minimum (0) primary star is eclipsed
        secondary_star_closer = np.logical_or(phases < 0.25, phases > 0.75)
        primary_star_overlap_ratio = np.where(
            secondary_star_closer, overlap_area / (np.pi * self.primary_radius**2), 0
        )
        # For phases around the secondary minimum (0.5) secondary star is eclipsed
        primary_star_closer = np.logical_and(phases >= 0.25, phases <= 0.75)
        secondary_star_overlap_ratio = np.where(
            primary_star_closer, overlap_area / (np.pi * self.secondary_radius**2), 0
        )

        primary_lum_density_eclipsed = primary_lum_density * (1 - primary_star_overlap_ratio)
        secondary_lum_density_eclipsed = secondary_lum_density * (1 - secondary_star_overlap_ratio)

        # Just in case, we clip negative values to zero
        total_lum_density = np.where(
            primary_lum_density_eclipsed >= 0, primary_lum_density_eclipsed, 0
        ) + np.where(secondary_lum_density_eclipsed >= 0, secondary_lum_density_eclipsed, 0)

        return total_lum_density

    @staticmethod
    def _norm_star_center_distance(phase_fraction, inclination_degree):
        """Calculate the distance between the centers of the stars on the plane of the sky.

        Parameters
        ----------
        phase_fraction : `np.ndarray`
            The phase of the orbit, in the range [0, 1].
        inclination_degree : `np.ndarray`
            The inclination of the orbit, in degrees.

        Returns
        -------
        distance : `no.ndarray`
            The distance between the centers of the stars on the plane of the sky,
            normalized by the major semiaxis.
        """
        phase_radians = 2 * np.pi * phase_fraction
        inclination_radians = np.radians(inclination_degree)
        return np.hypot(np.cos(inclination_radians) * np.cos(phase_radians), np.sin(phase_radians))

    # Math description can be found here:
    # https://math.stackexchange.com/a/3543551/1348501
    @staticmethod
    def _circle_overlap_area(d, r1, r2):
        """Calculate the area of overlap between two circles.

        Parameters
        ----------
        d : `np.ndarray`
            The distance between the centers of the circles.
        r1 : `float`
            The radius of the first circle.
        r2 : `float`
            The radius of the second circle.

        Returns
        -------
        overlap_area : `np.ndarray`
            The area of overlap between the two circles.
        """
        r1_sq = r1**2
        r2_sq = r2**2
        d_sq = d**2

        # Avoid invalid values for arccos and sqrt, we will handle the invalid values later
        with np.errstate(invalid="ignore"):
            term1 = r1_sq * np.arccos((d_sq + r1_sq - r2_sq) / (2 * d * r1))
            term2 = r2_sq * np.arccos((d_sq + r2_sq - r1_sq) / (2 * d * r2))
            term3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))

        area = term1 + term2 - term3
        # Circles do not overlap
        area = np.where(d >= r1 + r2, 0, area)
        # Circles fully overlap
        minimum_radius = np.minimum(r1, r2)
        area = np.where(d <= minimum_radius, np.pi * minimum_radius**2, area)

        return area


# We should move this function to a more appropriate location
def black_body_luminosity_density(temperature, radius, wavelengths):
    """Calculate the black-body luminosity density for a star.

    Parameters
    ----------
    temperature : `float`
        The effective temperature of the star, in kelvins.
    radius : `float`
        The radius of the star, in solar radii.
    wavelengths : `numpy.ndarray`
        A length N array of wavelengths.

    Returns
    -------
    luminosity_density : `numpy.ndarray`
        A length N array of luminosity density values.
    """
    black_body = BlackBody(temperature)
    intensity_per_freq = black_body(wavelengths * u.cm).to_cgs().value
    surface_flux = intensity_per_freq * np.pi
    surface_area = 4.0 * np.pi * radius**2
    return surface_flux * surface_area
