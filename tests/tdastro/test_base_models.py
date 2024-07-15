import random

import numpy as np
import pytest
from tdastro.base_models import FunctionNode, ParameterizedNode


def _sampler_fun(**kwargs):
    """Return a random value between 0 and 1.0.

    Parameters
    ----------
    **kwargs : `dict`, optional
        Absorbs additional parameters
    """
    return random.random()


def _test_func(value1, value2):
    """Return the sum of the two parameters.

    Parameters
    ----------
    value1 : `float`
        The first parameter.
    value2 : `float`
        The second parameter.
    """
    return value1 + value2


class PairModel(ParameterizedNode):
    """A test class for the ParameterizedNode.

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
        value1 : `float`, `function`, `ParameterizedNode`, or `None`
            The first value.
        value2 : `float`, `function`, `ParameterizedNode`, or `None`
            The second value.
        **kwargs : `dict`, optional
           Any additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_parameter("value1", value1, required=True, **kwargs)
        self.add_parameter("value2", value2, required=True, **kwargs)
        self.add_parameter("value_sum", self.result, required=True, **kwargs)

    def get_value1(self):
        """Get the value of value1."""
        return self.value1

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


def test_parameterized_node() -> None:
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


def test_parameterized_node_modify() -> None:
    """Test that we can modify the parameters in a node."""
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


def test_function_node_basic():
    """Test that we can create and query a FunctionNode."""
    my_func = FunctionNode(_test_func, value1=1.0, value2=2.0)
    assert my_func.compute() == 3.0
    assert my_func.compute(value2=3.0) == 4.0
    assert my_func.compute(value2=3.0, unused_param=5.0) == 4.0
    assert my_func.compute(value2=3.0, value1=1.0) == 4.0


def test_function_node_chain():
    """Test that we can create and query a chained FunctionNode."""
    func1 = FunctionNode(_test_func, value1=1.0, value2=1.0)
    func2 = FunctionNode(_test_func, value1=func1.compute, value2=3.0)
    assert func2.compute() == 5.0


def test_np_sampler_method():
    """Test that we can wrap numpy random functions."""
    rng = np.random.default_rng(1001)
    my_func = FunctionNode(rng.normal, loc=10.0, scale=1.0)

    # Sample 1000 times with the default values. Check that we are near
    # the expected mean and not everything is equal.
    vals = np.array([my_func.compute() for _ in range(1000)])
    assert abs(np.mean(vals) - 10.0) < 1.0
    assert not np.all(vals == vals[0])

    # Override the mean and resample.
    vals = np.array([my_func.compute(loc=25.0) for _ in range(1000)])
    assert abs(np.mean(vals) - 25.0) < 1.0
    assert not np.all(vals == vals[0])


def test_function_node_obj():
    """Test that we can create and query a FunctionNode that depends on
    another ParameterizedNode.
    """
    # The model depends on the function.
    func = FunctionNode(_test_func, value1=5.0, value2=6.0)
    model = PairModel(value1=10.0, value2=func.compute)
    assert model.result() == 21.0

    # Function depends on the model's value2 attribute.
    model = PairModel(value1=-10.0, value2=17.5)
    func = FunctionNode(_test_func, value1=5.0, value2=model)
    assert model.result() == 7.5
    assert func.compute() == 22.5

    # Function depends on the model's get_value1() method.
    func = FunctionNode(_test_func, value1=model.get_value1, value2=5.0)
    assert model.result() == 7.5
    assert func.compute() == -5.0

    # We can always override the attributes with kwargs.
    assert func.compute(value1=1.0, value2=4.0) == 5.0
