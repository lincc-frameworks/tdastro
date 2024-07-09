import numpy as np
import pytest
from numpy.testing import assert_allclose
from tdastro.sources.periodic_variable_star import EclipsingBinaryStar


def test_circle_overlap_area_total_eclipse():
    """Test that for r1=r2 and d=0, the overlap area is equal to the area of the circles"""
    r = 3.5
    d = 0.0
    overlap_area = EclipsingBinaryStar._circle_overlap_area(d, r, r)
    assert_allclose(overlap_area, np.pi * r**2)


def test_circle_overlap_area_primary_minimum():
    """Test that for r1>r2 and d < r1-r2, the overlap area is equal to the area of the secondary circle"""
    r1 = 3.5
    r2 = 1.5
    d = 1.0
    overlap_area = EclipsingBinaryStar._circle_overlap_area(d, r1, r2)
    assert_allclose(overlap_area, np.pi * r2**2)


def test_circle_overlap_area_no_overlap():
    """Test that for d>=r1+r2, the overlap area is zero"""
    r1 = 3.5
    r2 = 1.5
    d = r1 + r2
    overlap_area = EclipsingBinaryStar._circle_overlap_area(d, r1, r2)
    assert_allclose(overlap_area, 0.0)


@pytest.mark.parametrize(
    "d, r1, r2",
    [
        # Circle centers are inside circles.
        (1.0, 3.5, 3.0),
        # Intersections and secondary center lie on the same line.
        (3.0, 5.0, 4.0),
        # Center of the secondary is still inside the primary, but near the edge.
        (4.5, 5.0, 4.0),
        # Circle centers are outside each other.
        (5.0, 3.5, 3.0),
    ],
)
def test_circle_overlap_area_with_monte_carlo(d, r1, r2):
    """Test EclipsingBinaryStar._circle_overlap_area with MC simulation"""
    # The origin is the center of the primary star, the secondary star is at (d, 0).

    n_samples = 1_000_000
    rng = np.random.default_rng(0)

    # Generate random points in the bounding box
    x_range = [-r1, d + r2]
    assert r1 >= r2, "Check the test parameters, r1 should be not less than r2."
    y_range = [-r1, r1]
    x, y = rng.uniform([x_range[0], y_range[0]], [x_range[1], y_range[1]], (n_samples, 2)).T

    # Check if the points are inside the circles and find the overlap area
    inside_primary = np.hypot(x, y) < r1
    inside_secondary = np.hypot(x - d, y) < r2
    n_overlap = np.sum(inside_primary & inside_secondary)
    desired_overlap_area = n_overlap / n_samples * (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])

    actual_overlap_area = EclipsingBinaryStar._circle_overlap_area(d, r1, r2)
    # 2-sigma tolerance
    assert_allclose(actual_overlap_area, desired_overlap_area, rtol=2 / np.sqrt(n_overlap))
