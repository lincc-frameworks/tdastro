import numpy as np
from tdastro.sources.sncomso_models import SncosmoWrapperModel


def test_sncomso_models_hsiao() -> None:
    """Test that we can create and evalue a 'hsiao' model."""
    model = SncosmoWrapperModel("hsiao")
    model.set(amplitude=1.0e-10)
    assert model.amplitude == 1.0e-10
    assert str(model) == "tdastro.sources.sncomso_models.SncosmoWrapperModel"

    assert np.array_equal(model.param_names, ["amplitude"])
    assert np.array_equal(model.parameters, [1.0e-10])

    # Test with the example from: https://sncosmo.readthedocs.io/en/stable/models.html
    fluxes = model.evaluate([54990.0], [4000.0, 4100.0, 4200.0])
    assert np.allclose(fluxes, [4.31210900e-20, 7.46619962e-20, 1.42182787e-19])
