import numpy as np
import pytest
from lightcurvelynx.effects.effect_model import EffectModel


def test_effect_model() -> None:
    """The base effect model is able to extract the parameters."""
    model = EffectModel(param1=1.0, param2=2.0)
    assert "param1" in model.parameters
    assert "param2" in model.parameters
    assert model.parameters["param1"] == 1.0
    assert model.parameters["param2"] == 2.0
    assert model.effect_name == "EffectModel"
    assert str(model) == "EffectModel"
    assert repr(model) == "EffectModel(param1,param2)"

    # We can override the effect's name.
    model2 = EffectModel(param3=3.0, effect_name="test_effect")
    assert str(model2) == "test_effect"
    assert repr(model2) == "test_effect(param3)"

    # We cannot call apply on the base EffectModel
    with pytest.raises(NotImplementedError):
        _ = model.apply(np.zeros((5, 3)))
