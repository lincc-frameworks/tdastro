"""A wrapper class for representing detector footprints (which are stored as
astropy SkyRegions). This class provides methods for checking if points are within
the footprint and for plotting the footprint."""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS


class DetectorFootprint:
    """A wrapper class for representing detector footprints

    Attributes
    ----------
    region : astropy.regions.SkyRegion
        The astropy SkyRegion representing the footprint.
    wcs : astropy.wcs.WCS or None
        The WCS associated with the region, if any.
    pixel_scale : float or None
        The pixel scale in degrees/pixel, this is required if no WCS is provided.
    """

    def __init__(self, region, wcs=None, pixel_scale=None, **kwargs):
        self.region = region
        self.pixel_scale = pixel_scale

        # Create a default WCS if none is provided.
        if wcs is None:
            if pixel_scale is None:
                raise ValueError("Either wcs or pixel_scale must be provided.")
            if pixel_scale <= 0:
                raise ValueError("pixel_scale must be positive.")

            # Create a simple TAN WCS centered on (0.0, 0.0) with the given pixel scale.
            wcs = WCS(naxis=2)
            wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            wcs.wcs.crval = [0.0, 0.0]  # Centered on (0.0, 0.0)
            wcs.wcs.crpix = [0.0, 0.0]  # Reference pixel at (0.0, 0.0)
            wcs.wcs.cdelt = [pixel_scale, pixel_scale]  # The given pixel scale in degrees/pixel
        self.wcs = wcs

    @staticmethod
    def rotate_to_center(ra, dec, center_ra, center_dec, *, rotation=None):
        """Transform the given points, represented by (ra, dec) in degrees,
        to a local coordinate system centered on (center_ra, center_dec), accounting
        for rotation if specified.

        Note
        ----
        This method is vectorized so it can handle an array of points and an (equally
        sized) array of center points. The transformation is done on each point for the
        corresponding center point.

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
            Transformed (lon, lat) offsets from the center in degrees.
        """
        if rotation is None:
            rotation = np.zeros_like(ra)

        target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        origin = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg, frame="icrs")
        offset_frame = origin.skyoffset_frame(rotation=rotation * u.deg)
        offset = target.transform_to(offset_frame)

        return offset.lon.deg, offset.lat.deg

    def contains(self, ra, dec, center_ra, center_dec, *, rotation=None):
        """Check that given points, represented by (ra, dec) in degrees,
        are within the footprint.

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

        # Transform the (ra, dec) to local coordinates centered on (center_ra, center_dec).
        d_ra, d_dec = self._transform(ra, dec, center_ra, center_dec, rotation=rotation)

        # Check if each point is within the footprint.
        skycoord = SkyCoord(d_ra, d_dec, unit="deg")
        result = self.region.contains(skycoord, self.wcs)

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

        ax.set_title("Detector Footprint")
        ax.axis("equal")
        return ax
