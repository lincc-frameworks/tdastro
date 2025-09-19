"""A class for representing survey footprints."""

import matplotlib.pyplot as plt
import numpy as np


class SurveyFootprint:
    """A base class for representing survey footprints. This class is a No-Op and
    does not implement any filtering.
    """

    def __init__(self, **kwargs):
        pass

    def _transform(self, ra, dec, center_ra, center_dec, *, rotation=None):
        """Transform the given points, represented by (ra, dec) in degrees,
        to a local coordinate system centered on (center_ra, center_dec), accounting
        for rotation if specified.

        Note
        ----
        This method assumes small angles and does not account for spherical geometry.
        For large footprints or high precision, a more accurate transformation should
        be implemented.

        Parameters
        ----------
        ra : np.ndarray
            Right ascension in degrees.
        dec : np.ndarray
            Declination in degrees.
        center_ra : np.ndarray
            Center right ascension of the detector in degrees.
        center_dec : np.ndarray
            Center declination of the detector in degrees.
        rotation : np.ndarray, optional
            The rotation angle of the detector for each pointing in degrees clockwise.
            Used to represent non-axis-aligned footprints.

        Returns
        -------
        tuple of np.ndarray
            Transformed (x, y) coordinates in radians.
        """
        # Compute the distance in radians from the center of the footprint,
        # accounting for RA wrapping.
        ra_diff = np.radians(ra - center_ra)
        ra_diff[ra_diff > np.pi] -= 2 * np.pi
        ra_diff[ra_diff < -np.pi] += 2 * np.pi
        dec_diff = np.radians(dec - center_dec)

        # If rotation is specified, rotate the coordinates accordingly. We rotate
        # counter-clockwise to account for the clockwise rotation of the footprint.
        if rotation is not None:
            rotation_rad = np.radians(rotation)
            cos_rot = np.cos(rotation_rad)
            sin_rot = np.sin(rotation_rad)
            ra_rot = ra_diff * cos_rot - dec_diff * sin_rot
            dec_rot = ra_diff * sin_rot + dec_diff * cos_rot
            ra_diff, dec_diff = ra_rot, dec_rot

        return ra_diff, dec_diff

    def _contains(self, d_x, d_y):
        """Check that given points, represented by distance in radians from the center
        (accounting for rotation if specified), are within the footprint. This is a
        No-Op implementation that always returns True. Subclasses should override this
        method to implement specific footprint logic.

        Parameters
        ----------
        d_x : np.ndarray
            Distance in rotated x coordinate from the center of the detector in radians.
        d_y : np.ndarray
            Distance in rotated y coordinate from the center of the detector in radians.

        Returns
        -------
        np.ndarray
            True if the point is within the footprint, False otherwise.
        """
        return np.full(np.shape(d_x), True, dtype=bool)

    def contains(self, ra, dec, center_ra, center_dec, *, rotation=None):
        """Check that given points, represented by (ra, dec) in degrees,
        are within the footprint. This is the outer interface that handles
        both scalar and array inputs, does validation, and delegates to the
        internal implementation. Subclasses should override _contains.

        Parameters
        ----------
        ra : float or array-like
            Right ascension in degrees.
        dec : float or array-like
            Declination in degrees.
        center_ra : float or array-like
            Center right ascension of the detector in degrees.
        center_dec : float or array-like
            Center declination of the detector in degrees.
        rotation : np.ndarray, optional
            The rotation angle of the detector for each pointing in degrees clockwise.
            Used to represent non-axis-aligned footprints.

        Returns
        -------
        bool or array-like of bool
            True if the point is within the footprint, False otherwise.
        """
        # Convert all inputs to numpy arrays for uniform processing and perform validation.
        scalar_data = np.isscalar(ra)
        ra = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        center_ra = np.atleast_1d(center_ra)
        center_dec = np.atleast_1d(center_dec)
        if ra.shape != dec.shape:
            raise ValueError("ra and dec must have the same shape.")
        if center_ra.shape != ra.shape:
            raise ValueError("center_ra must have the same shape as ra and dec.")
        if center_dec.shape != dec.shape:
            raise ValueError("center_dec must have the same shape as ra and dec.")

        # Rotation is optional, but if provided, it must match the shape of ra and dec.
        if rotation is not None:
            rotation = np.atleast_1d(rotation)
            if rotation.shape != ra.shape:
                print(rotation.shape)
                print(ra.shape)
                raise ValueError("rotation must have the same shape as ra and dec.")

        d_x, d_y = self._transform(ra, dec, center_ra, center_dec, rotation=rotation)
        result = self._contains(d_x, d_y)
        if scalar_data:
            return result[0]
        return result

    def plot_bounds(self, ax, center_ra=0, center_dec=0, rotation=0, **kwargs):
        """Plot the bounds of the footprint on the given axes. This base implementation
        does nothing. Subclasses should override this method to implement specific plotting
        logic.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            Axes to plot on.
        center_ra : float, optional
            Center right ascension of the detector in degrees. Default is 0.
        center_dec : float, optional
            Center declination of the detector in degrees. Default is 0.
        rotation : np.ndarray, optional
            The rotation angle of the detector for each pointing in degrees clockwise.
            Used to represent non-axis-aligned footprints.
            Default is 0.
        **kwargs : dict
            Optional parameters to pass to the plotting function
        """
        pass

    def plot(
        self,
        *,
        ax=None,
        figure=None,
        center_ra=0,
        center_dec=0,
        rotation=0,
        point_ra=None,
        point_dec=None,
        **kwargs,
    ):
        """Plot the footprint using matplotlib and an optional set of points to overlay.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes or None, optional
            Axes, If None, a new axes will be created. None by default.
        figure : matplotlib.pyplot.Figure or None
            Figure, If None, a new figure will be created. None by default.
        center_ra : float, optional
            Center right ascension of the detector in degrees. Default is 0.
        center_dec : float, optional
            Center declination of the detector in degrees. Default is 0.
        rotation : np.ndarray, optional
            The rotation angle of the detector for each pointing in degrees clockwise.
            Used to represent non-axis-aligned footprints.
            Default is 0.
        point_ra : array-like or None, optional
            Right ascension of points to overlay in degrees. None by default.
        point_dec : array-like or None, optional
            Declination of points to overlay in degrees. None by default.
        **kwargs : dict
            Optional parameters to pass to the plotting function

        Returns
        -------
        ax : matplotlib.pyplot.Axes
            The axes containing the plot.
        """
        if ax is None:
            if figure is None:
                figure = plt.figure()
            ax = figure.add_axes([0, 0, 1, 1])

        # Plot the bounds of the footprint.
        self.plot_bounds(ax, center_ra=center_ra, center_dec=center_dec, rotation=rotation, **kwargs)

        # If points are provided, overlay them on the plot.
        if point_ra is not None and point_dec is not None:
            point_ra = np.asarray(point_ra)
            point_dec = np.asarray(point_dec)

            center_ra = np.full(np.shape(point_ra), center_ra)
            center_dec = np.full(np.shape(point_dec), center_dec)
            if rotation is not None:
                rotation = np.full(np.shape(point_ra), rotation)

            isin = self.contains(point_ra, point_dec, center_ra, center_dec, rotation=rotation)
            ax.scatter(point_ra[~isin], point_dec[~isin], color="red", marker="x", label="Outside Footprint")
            ax.scatter(point_ra[isin], point_dec[isin], color="green", marker="o", label="Inside Footprint")
            ax.legend()

        ax.set_title("Survey Footprint")
        ax.axis("equal")
        return ax


class CircularFootprint(SurveyFootprint):
    """A circular survey footprint defined by a radius.

    Attributes
    ----------
    radius_rad : float
        Radius of the circle in radians.

    Parameters
    ----------
    radius : float
        Radius of the circle in degrees.
    """

    def __init__(self, radius):
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self.radius_rad = np.radians(radius)

    def _contains(self, d_x, d_y):
        """Check that given points, represented by distance in radians from the center
        (accounting for rotation if specified), are within the footprint.

        Parameters
        ----------
        d_x : np.ndarray
            Distance in rotated x coordinate from the center of the detector in radians.
        d_y : np.ndarray
            Distance in rotated y coordinate from the center of the detector in radians.

        Returns
        -------
        np.ndarray
            True if the point is within the footprint, False otherwise.
        """
        return (d_x**2 + d_y**2) <= self.radius_rad**2

    def plot_bounds(self, ax, center_ra=0, center_dec=0, rotation=0, **kwargs):
        """Plot the bounds of the footprint on the given axes.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            Axes to plot on.
        center_ra : float, optional
            Center right ascension of the detector in degrees. Default is 0.
        center_dec : float, optional
            Center declination of the detector in degrees. Default is 0.
        rotation : np.ndarray, optional
            Not used for circular footprints, but included for interface consistency.
            Default is 0.
        **kwargs : dict
            Optional parameters to pass to the plotting function
        """
        radius = np.degrees(self.radius_rad)
        circle = plt.Circle(
            (center_ra, center_dec),
            radius,
            color="k",
            fill=False,
            **kwargs,
        )
        ax.add_artist(circle)
        ax.set_xlim((center_ra - 2 * radius, center_ra + 2 * radius))
        ax.set_ylim((center_dec - 2 * radius, center_dec + 2 * radius))


class RectangularFootprint(SurveyFootprint):
    """A rectangular survey footprint defined by minimum and maximum
    right ascension and declination.

    Attributes
    ----------
    height : float
        Height of the rectangle in degrees.
    width : float
        Width of the rectangle in degrees.
    half_height_rad : float
        Half the height of the rectangle in radians.
    half_width_rad : float
        Half the width of the rectangle in radians.

    Parameters
    ----------
    height : float
        Height of the rectangle in degrees.
    width : float
        Width of the rectangle in degrees.
    """

    def __init__(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive.")
        self.half_width_rad = np.radians(width) / 2
        self.half_height_rad = np.radians(height) / 2

    def _contains(self, d_x, d_y):
        """Check that given points, represented by distance in radians from the center
        (accounting for rotation if specified), are within the footprint. This is a
        No-Op implementation that always returns True. Subclasses should override this
        method to implement specific footprint logic.

        Parameters
        ----------
        d_x : np.ndarray
            Distance in rotated x coordinate from the center of the detector in radians.
        d_y : np.ndarray
            Distance in rotated y coordinate from the center of the detector in radians.

        Returns
        -------
        np.ndarray
            True if the point is within the footprint, False otherwise.
        """
        return (np.abs(d_x) <= self.half_width_rad) & (np.abs(d_y) <= self.half_height_rad)

    def plot_bounds(self, ax, center_ra=0, center_dec=0, rotation=0, **kwargs):
        """Plot the bounds of the footprint on the given axes. This base implementation
        does nothing. Subclasses should override this method to implement specific plotting
        logic.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            Axes to plot on.
        center_ra : float, optional
            Center right ascension of the detector in degrees. Default is 0.
        center_dec : float, optional
            Center declination of the detector in degrees. Default is 0.
        rotation : np.ndarray, optional
            The rotation angle of the detector for each pointing in degrees clockwise.
            Used to represent non-axis-aligned footprints.
            Default is 0.
        **kwargs : dict
            Optional parameters to pass to the plotting function
        """
        # Define the corners of the rectangle in local coordinates (radians).
        half_w = self.half_width_rad
        half_h = self.half_height_rad
        corners_x = np.array([-half_w, half_w, half_w, -half_w, -half_w])
        corners_y = np.array([-half_h, -half_h, half_h, half_h, -half_h])

        # If rotation is specified, rotate the corners accordingly.
        if rotation:
            rotation_rad = np.radians(rotation)
            cos_rot = np.cos(rotation_rad)
            sin_rot = np.sin(rotation_rad)
            x_rot = corners_x * cos_rot + corners_y * sin_rot
            y_rot = -corners_x * sin_rot + corners_y * cos_rot
            corners_x, corners_y = x_rot, y_rot

        # Convert the corners back to RA/Dec coordinates.
        corners_ra = np.degrees(corners_x) + center_ra
        corners_dec = np.degrees(corners_y) + center_dec

        # Plot the rectangle.
        ax.plot(corners_ra, corners_dec, color="k", **kwargs)
