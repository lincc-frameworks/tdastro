import numpy as np
from lightcurvelynx.astro_utils.detector_footprint import DetectorFootprint


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

    # Test with rotation. Note the points will be rotated counter-clockwise
    # since the footprint is rotated clockwise.
    # rotation = np.array([0.0, 90.0, 180.0])
    # ra_t, dec_t = DetectorFootprint.rotate_to_center(
    #    ra, dec, center_ra, center_dec, rotation=rotation
    # )
    # assert np.allclose(ra_t, [0.0, 1.0, 0.0])
    # assert np.allclose(dec_t, [0.0, 1.0, 2.0])
