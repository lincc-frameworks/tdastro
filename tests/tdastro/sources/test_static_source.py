import numpy as np
from tdastro.common_citations import numpy_citation
from tdastro.sources.static_source import StaticSource


def test_static_source() -> None:
    """Test that we can sample and create a StaticSource object."""
    model = StaticSource(brightness=10.0)
    assert model.brightness == 10.0
    assert model.ra is None
    assert model.dec is None
    assert model.distance is None

    times = np.array([1, 2, 3, 4, 5, 10])
    wavelengths = np.array([100.0, 200.0, 300.0])

    values = model.evaluate(times, wavelengths)
    assert values.shape == (6, 3)
    assert np.all(values == 10.0)


def test_static_source_host() -> None:
    """Test that we can sample and create a StaticSource object with properties
    derived from the host object."""
    host = StaticSource(brightness=15.0, ra=1.0, dec=2.0, distance=3.0)
    model = StaticSource(brightness=10.0, host=host)
    assert model.brightness == 10.0
    assert model.ra == 1.0
    assert model.dec == 2.0
    assert model.distance == 3.0


def test_static_source_citations() -> None:
    """Test that we get the citation string for numpy."""
    model = StaticSource(brightness=10.0)
    citations = model.get_citations()
    assert len(citations) == 1
    assert numpy_citation in citations
