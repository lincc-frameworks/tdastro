import random

import pytest
from tdastro.base_models import ParameterizedModel


def _sampler_fun(**kwargs):
    """Return a random value between 0 and 1.0.

    Parameters
    ----------
    **kwargs : `dict`, optional
        Absorbs additional parameters
    """
    return random.random()


class PairModel(ParameterizedModel):
    """A test class for the parameterized model.

    Attributes
    ----------
    value1 : `float`
        The first value.
    value2 : `float`
        The second value.
    value_sum : `float`
        The sum of the two values.
    """

    def __init__(self, value1, value2, **kwargs):
        """Create a ConstModel object.

        Parameters
        ----------
        value1 : `float`, `function`, `ParameterizedModel`, or `None`
            The first value.
        value2 : `float`, `function`, `ParameterizedModel`, or `None`
            The second value.
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_parameter("value1", value1, required=True, **kwargs)
        self.add_parameter("value2", value2, required=True, **kwargs)
        self.add_parameter("value_sum", self.result, required=True, **kwargs)

    def result(self, **kwargs):
        """Add the pair of values together

        Parameters
        ----------
        **kwargs : `dict`, optional
            Any additional keyword arguments.

        Returns
        -------
        result : `float`
            The result of the addition.
        """
        return self.value1 + self.value2


def test_parameterized_model() -> None:
    """Test that we can sample and create a PairModel object."""
    # Simple addition
    model1 = PairModel(value1=0.5, value2=0.5)
    assert model1.value1 == 0.5
    assert model1.value1 == 0.5
    assert model1.result() == 1.0
    assert model1.value_sum == 1.0
    assert model1.sample_iteration == 0

    # Use value1=model.value and value2=1.0
    model2 = PairModel(value1=model1, value2=1.0)
    assert model2.value1 == 0.5
    assert model2.value2 == 1.0
    assert model2.result() == 1.5
    assert model2.value_sum == 1.5
    assert model2.sample_iteration == 0

    # Compute value1 from model2's result and value2 from the sampler function.
    model3 = PairModel(value1=model2.result, value2=_sampler_fun)
    rand_val = model3.value2
    assert model3.result() == pytest.approx(1.5 + rand_val)
    assert model3.value_sum == pytest.approx(1.5 + rand_val)
    assert model3.sample_iteration == 0

    # Compute value1 from model3's result (which is itself the result for model2 +
    # a random value) and value2 = -1.0.
    model4 = PairModel(value1=model3.result, value2=-1.0)
    assert model4.result() == pytest.approx(0.5 + rand_val)
    assert model4.value_sum == pytest.approx(0.5 + rand_val)
    assert model4.sample_iteration == 0
    final_res = model4.result()

    # We can resample and it should change the result.
    while model3.value2 == rand_val:
        print(f"{model3.value2}")
        model4.sample_parameters()
    rand_val = model3.value2

    # Nothing changes in model1 or model2
    assert model1.value1 == 0.5
    assert model1.value2 == 0.5
    assert model1.result() == 1.0
    assert model1.value_sum == 1.0
    assert model2.value1 == 0.5
    assert model2.value2 == 1.0
    assert model2.result() == 1.5
    assert model2.value_sum == 1.5

    # Models 3 and 4 use the data from the new random value.
    assert model3.result() == pytest.approx(1.5 + rand_val)
    assert model4.result() == pytest.approx(0.5 + rand_val)
    assert model3.value_sum == pytest.approx(1.5 + rand_val)
    assert model4.value_sum == pytest.approx(0.5 + rand_val)
    assert final_res != model4.result()

    # All models should have the same sample iteration.
    assert model1.sample_iteration == model2.sample_iteration
    assert model1.sample_iteration == model3.sample_iteration
    assert model1.sample_iteration == model4.sample_iteration


def test_parameterized_model_modify() -> None:
    """Test that we can modify the parameters in a model."""
    model = PairModel(value1=0.5, value2=0.5)
    assert model.value1 == 0.5
    assert model.value2 == 0.5

    # We cannot add a parameter a second time.
    with pytest.raises(KeyError):
        model.add_parameter("value1", 5.0)

    # We can set the parameter.
    model.set_parameter("value1", 5.0)
    assert model.value1 == 5.0
    assert model.value2 == 0.5

    # We cannot set a value that hasn't been added.
    with pytest.raises(KeyError):
        model.set_parameter("brightness", 5.0)
