import numpy as np
import pytest
from tdastro.effects.effect_model import EffectModel


def test_effect_model() -> None:
    """The base effect model is able to extract the parameters."""
    model = EffectModel(param1=1.0, param2=2.0)
    assert "param1" in model.parameters
    assert "param2" in model.parameters
    assert model.parameters["param1"] == 1.0
    assert model.parameters["param2"] == 2.0

    # We cannot call apply on the base EffectModel
    with pytest.raises(NotImplementedError):
        _ = model.apply(np.zeros((5, 3)))
