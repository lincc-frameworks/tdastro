from tdastro.sources.physical_model import PhysicalModel


def test_physical_model():
    """Test that we can create a PhysicalModel."""
    # Everything is specified.
    model1 = PhysicalModel(ra=1.0, dec=2.0, distance=3.0)
    assert model1.ra == 1.0
    assert model1.dec == 2.0
    assert model1.distance == 3.0
