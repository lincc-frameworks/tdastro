"""A wrapper class for representing detector footprints (which are stored as
astropy SkyRegions). This class provides methods for checking if points are within
the footprint and for plotting the footprint."""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from regions import PixCoord, RectangleSkyRegion, SkyRegion


class DetectorFootprint:
    """A wrapper class for representing detector footprints.

    Attributes
    ----------
    region : astropy.regions.SkyRegion or Astropy.regions.PixelRegion
        The astropy SkyRegion or PixelRegion representing the footprint.
    wcs : astropy.wcs.WCS or None
        The WCS associated with the region, if any.
    pixel_scale : float or None
        The pixel scale in degrees/pixel, this is required if no WCS is provided.
    """

    def __init__(self, region, wcs=None, pixel_scale=None, **kwargs):
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
            wcs.wcs.crpix = [0.5, 0.5]  # Reference pixel at the center of (0.0, 0.0)
            wcs.wcs.cdelt = [pixel_scale, pixel_scale]  # The given pixel scale in degrees/pixel
        self.wcs = wcs

        # Store the region as a pixel region, since we will always need to do a conversion
        # as part of the contains method otherwise.
        self.region = region
        if isinstance(self.region, SkyRegion):
            self.region = self.region.to_pixel(self.wcs)

    @classmethod
    def from_rect(cls, width, height, wcs=None, pixel_scale=None, **kwargs):
        """Create a rectangular footprint.

        Parameters
        ----------
        width : float
            Width of the rectangle in degrees.
        height : float
            Height of the rectangle in degrees.
        wcs : astropy.wcs.WCS, optional
            The WCS associated with the region. If None, a default WCS will be created.
        pixel_scale : float, optional
            The pixel scale in degrees/pixel, this is required if no WCS is provided.
        **kwargs : dict
            Additional keyword arguments to pass to the RectangleSkyRegion constructor.

        Returns
        -------
        DetectorFootprint
            The rectangular detector footprint.
        """
        center = SkyCoord(ra=0.0, dec=0.0, unit="deg", frame="icrs")
        region = RectangleSkyRegion(
            center=center,
            width=width * u.deg,
            height=height * u.deg,
            angle=0.0 * u.deg,
            **kwargs,
        )
        return cls(region, wcs=wcs, pixel_scale=pixel_scale)

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

        # Center everything on 0.0
        lon_t = offset.lon.deg
        lon_t[lon_t > 180.0] -= 360.0

        lat_t = offset.lat.deg
        lat_t[lat_t > 90.0] -= 180.0

        return lon_t, lat_t

    def sky_to_pixel(self, ra, dec, center_ra, center_dec, *, rotation=None):
        """Transform sky coordinates (ra, dec) to pixel coordinates.

        Parameters
        ----------
        ra : float or array-like
            Right ascension in degrees.
        dec : float or array-like
            Declination in degrees.
        center_ra : float
            Center right ascension of the detector in degrees.
        center_dec : float
            Center declination of the detector in degrees.
        rotation : float or array-like, optional
            The rotation angle of the detector for each pointing in degrees clockwise.
            Used to represent non-axis-aligned footprints.

        Returns
        -------
        tuple of np.ndarray
            Pixel coordinates (x, y) corresponding to the input sky coordinates.
        """
        lon_t, lat_t = self.rotate_to_center(ra, dec, center_ra, center_dec, rotation=rotation)
        sky_pts = SkyCoord(lon_t, lat_t, unit="deg")
        pixel_x, pixel_y = self.wcs.world_to_pixel(sky_pts)
        return pixel_x, pixel_y

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
        scalar_data = np.isscalar(ra)

        # Rotate the points to be relative to the center and convert to pixel coordinates.
        pixel_x, pixel_y = self.sky_to_pixel(ra, dec, center_ra, center_dec, rotation=rotation)
        pix_coord = PixCoord(x=pixel_x, y=pixel_y)

        # Check if each point is within the footprint.
        result = self.region.contains(pix_coord)
        if scalar_data:
            return result[0]
        return result

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
        """
        if ax is None:
            if figure is None:
                figure = plt.figure()
            ax = figure.add_axes([0, 0, 1, 1])

        # Plot the bounds of the footprint.
        artist = self.region.as_artist()
        ax.add_artist(artist)

        # Get the bounding box in pixel coordinates. Expand out to ensure the full
        # footprint is visible.
        bbox = self.region.bounding_box
        width = bbox.ixmax - bbox.ixmin
        height = bbox.iymax - bbox.iymin
        xmin = bbox.ixmin - 0.5 * width
        xmax = bbox.ixmax + 0.5 * width
        ymin = bbox.iymin - 0.5 * height
        ymax = bbox.iymax + 0.5 * height

        # If points are provided, overlay them on the plot.
        if point_ra is not None and point_dec is not None:
            # Compute the transformed pixel coordinates relative to the center.
            pixel_x, pixel_y = self.sky_to_pixel(
                point_ra, point_dec, center_ra, center_dec, rotation=rotation
            )

            # Since we have already transformed the points to be relative to the center,
            # we can check if they are within the footprint without further transformation.
            isin = self.region.contains(PixCoord(x=pixel_x, y=pixel_y))

            ax.scatter(pixel_x[~isin], pixel_y[~isin], color="red", marker="x", label="Outside Footprint")
            ax.scatter(pixel_x[isin], pixel_y[isin], color="green", marker="o", label="Inside Footprint")
            ax.legend()

            # Adjust the bounding box to include all the points.
            if xmin < pixel_x.min():
                xmin = pixel_x.min() - 2.0
            if xmax > pixel_x.max():
                xmax = pixel_x.max() + 2.0
            if ymin < pixel_y.min():
                ymin = pixel_y.min() - 2.0
            if ymax > pixel_y.max():
                ymax = pixel_y.max() + 2.0

        ax.set_title("Detector Footprint")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")
        ax.set_aspect("equal")
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        plt.show()
