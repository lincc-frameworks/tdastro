import numpy as np
import pytest
from lightcurvelynx.astro_utils.detector_footprint import (
    CircularFootprint,
    RectangularFootprint,
    DetectorFootprint,
)


def test_detector_footprint_base_class():
    """Test the base DetectorFootprint class."""
    fp = DetectorFootprint()

    ra = np.array([90.0, 91.0, 92.0])
    dec = np.array([-10.0, -11.0, -10.0])
    center_ra = np.array([90.0, 90.0, 92.0])
    center_dec = np.array([-10.0, -10.0, -8.0])

    # Test the internal transform method. The points should be relative to the
    # center and in radians.
    ra_t, dec_t = fp._transform(ra, dec, center_ra, center_dec)
    assert np.allclose(ra_t, np.radians([0.0, 1.0, 0.0]))
    assert np.allclose(dec_t, np.radians([0.0, -1.0, -2.0]))

    # Test with rotation. Note the points will be rotated counter-clockwise
    # since the footprint is rotated clockwise.
    rotation = np.array([0.0, 90.0, 180.0])
    ra_t, dec_t = fp._transform(ra, dec, center_ra, center_dec, rotation=rotation)
    assert np.allclose(ra_t, np.radians([0.0, 1.0, 0.0]))
    assert np.allclose(dec_t, np.radians([0.0, 1.0, 2.0]))

    # Default contains should always return True for the base class.
    assert np.all(fp.contains(ra, dec, center_ra, center_dec))
    assert np.all(fp.contains(ra, dec, center_ra, center_dec, rotation=rotation))

    # We can query scalars as well.
    assert fp.contains(90.0, -10.0, 90.0, -10.0)
    assert fp.contains(90.0, -10.0, 90.0, -10.0, rotation=90.0)

    # We fail if the arrays are not the same shape.
    with pytest.raises(ValueError):
        fp.contains(ra, 1.0, center_ra, center_dec)
    with pytest.raises(ValueError):
        fp.contains(ra, dec, 1.0, center_dec)
    with pytest.raises(ValueError):
        fp.contains(ra, dec, center_ra, 1.0)
    with pytest.raises(ValueError):
        fp.contains(ra, dec, center_ra, center_dec, rotation=1.0)


def test_circular_footprint():
    """Test the CircularFootprint class."""
    fp = CircularFootprint(1.0)

    ra = np.array([90.0, 95.0, 90.0, 90.4])
    center_ra = np.array([90.0, 90.0, 92.0, 90.0])
    dec = np.array([-10.0, -15.0, -10.0, 10.4])
    center_dec = np.array([-10.0, -10.0, -8.0, 10.0])

    contained = fp.contains(ra, dec, center_ra, center_dec)
    assert np.all(contained == np.array([True, False, False, True]))

    # We can call the plot function without and with points. Here we are just testing
    # that nothing crashes. The visualization can be confirmed in the notebook.
    _ = fp.plot(center_ra=90.0, center_dec=-10.0, rotation=45.0)


def test_rectangular_footprint():
    """Test the RectangularFootprint class."""
    width = 2.0
    height = 1.0
    fp = RectangularFootprint(width=width, height=height)

    ra = np.array([90.0, 91.0, 92.0, 90.5, 90.0])
    center_ra = np.array([90.0, 90.0, 92.0, 93.0, 90.0])
    dec = np.array([-10.0, -13.0, -8.0, -10.25, -9.25])
    center_dec = np.array([-10.0, -10.0, -8.0, -10.0, -10.0])

    # Test contains without rotation.
    contained = fp.contains(ra, dec, center_ra, center_dec)
    assert np.all(contained == np.array([True, False, True, False, False]))

    # Test contains with rotation. The fifth point should now be contained.
    rotation = np.full(ra.shape, 90.0)
    contained = fp.contains(ra, dec, center_ra, center_dec, rotation=rotation)
    print(contained)
    assert np.all(contained == np.array([True, False, True, False, True]))

    # We can query scalars as well.
    assert fp.contains(90.5, -10.25, 90.0, -10.0)
    assert not fp.contains(91.5, -10.25, 90.0, -10.0)
    assert fp.contains(89.75, -9.75, 90.0, -10.0)
    assert not fp.contains(89.75, -9.25, 90.0, -10.0)

    assert not fp.contains(91.5, -10.25, 90.0, -10.0, rotation=45.0)
    assert fp.contains(89.75, -9.75, 90.0, -10.0, rotation=-45.0)
    assert not fp.contains(89.75, -9.25, 90.0, -10.0, rotation=-45.0)

    # We can call the plot function without and with points. Here we are just testing
    # that nothing crashes. The visualization can be confirmed in the notebook.
    _ = fp.plot()
    _ = fp.plot(center_ra=90.0, center_dec=-10.0, rotation=45.0)
    _ = fp.plot(
        center_ra=90.0,
        center_dec=-10.0,
        rotation=45.0,
        point_ra=np.array([90.0, 91.0]),
        point_dec=np.array([-10.0, -10.5]),
    )
