import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from lightcurvelynx.astro_utils.detector_footprint import DetectorFootprint
from regions import CirclePixelRegion, CircleSkyRegion


def test_rotate_to_center():
    """Test the static rotate_to_center method."""
    # Define the test data as a numpy array with one row for each test case,
    # and columns for ra, dec, center_ra, center_dec, expected_ra, expected_dec.
    tests = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # No shift
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Small shift in RA only
            [57.0, 0.0, 0.0, 0.0, 57.0, 0.0],  # Large shift in RA only
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Shift in Dec
            [91.0, -10.0, 91.0, -10.0, 0.0, 0.0],  # Shift to same point
            [45.0, -89.0, 45.0, -90.0, 0.0, 1.0],  # Same RA, different DEC
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],  # Shift in both
            [45.0, 45.0, 40.0, 40.0, 3.54734593, 5.09948400],  # Large shift in both
        ]
    )

    # Test the internal transform method. The points should be relative to the
    # center and in radians.
    lon_t, lat_t = DetectorFootprint.rotate_to_center(tests[:, 0], tests[:, 1], tests[:, 2], tests[:, 3])
    assert np.allclose(lon_t, tests[:, 4], atol=1e-6)
    assert np.allclose(lat_t, tests[:, 5], atol=1e-6)

    # Perform some additional tests with rotation. The test data is now
    # ra, dec, center_ra, center_dec, rotation, expected_ra, expected_dec
    tests = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 90.0, 0.0, 1.0],  # Rotate RA=1 to dec=1
            [0.0, 1.0, 0.0, 0.0, 90.0, -1.0, 0.0],  # Rotate dec=1 to RA=-1
            [1.0, 0.0, 0.0, 0.0, 45.0, 0.7071, 0.7071],  # Rotate RA=1 to RA=0.7,dec=0.7
            [1.0, 1.0, 0.0, 0.0, 90.0, -1.0, 1.0],  # Rotate RA=1,dec=1 to RA=-1,dec=1
            [1.0, 1.0, 0.0, 0.0, -45.0, 1.4142, 0.0],  # Rotate RA=1,dec=1 to RA=1.4,dec=0
            [45.0, 45.0, 40.0, 40.0, 45.0, -1.111, 6.1095],  # Large shift in both with rotation
        ]
    )

    # Test the internal transform method.
    lon_t, lat_t = DetectorFootprint.rotate_to_center(
        tests[:, 0],
        tests[:, 1],
        tests[:, 2],
        tests[:, 3],
        rotation=tests[:, 4],
    )
    assert np.allclose(lon_t, tests[:, 5], atol=1e-3)
    assert np.allclose(lat_t, tests[:, 6], atol=1e-3)


def test_create_detector_footprint():
    """Test creating a DetectorFootprint."""
    center = SkyCoord(ra=0.0, dec=0.0, unit="deg", frame="icrs")
    circle_region = CircleSkyRegion(center=center, radius=1.0 * u.deg)
    fp = DetectorFootprint(circle_region, pixel_scale=0.000277778)  # 1 arcsec/pixel
    assert isinstance(fp, DetectorFootprint)
    assert isinstance(fp.region, CirclePixelRegion)
    assert fp.wcs is not None

    # Test that the contains method works.
    ra = np.array([0.0, 0.5, 1.0, 1.5, 0.0, 1.5, 0.9])  # degrees
    dec = np.array([0.0, 0.5, 1.0, 1.5, 1.5, 0.0, 0.0])  # degrees
    result = fp.contains(ra, dec, center_ra=0.0, center_dec=0.0)
    expected = np.array([True, True, False, False, False, False, True])
    assert np.array_equal(result, expected)

    # We can try a circular region that is offset from the center.
    result = fp.contains(ra + 45.0, dec - 10.0, center_ra=45.0, center_dec=-10.0)
    expected = np.array([True, True, False, False, False, False, True])
    assert np.array_equal(result, expected)
