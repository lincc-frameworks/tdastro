import numpy as np
import pytest
from tdastro.function_wrappers import TDFunc


def _test_func(a, b):
    """Return the sum of the two parameters.

    Parameters
    ----------
    a : `float`
        The first parameter.
    b : `float`
        The second parameter.
    """
    return a + b


class _StaticModel:
    """A test model that has given parameters.

    Attributes
    ----------
    a : `float`
        The first parameter.
    b : `float`
        The second parameter.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def eval_func(self, func, **kwargs):
        """Evaluate a TDFunc.

        Parameters
        ----------
        func : `TDFunc`
            The function to evaluate.
        **kwargs : `dict`, optional
            Any additional keyword arguments.
        """
        return func(self, **kwargs)


def test_tdfunc_basic():
    """Test that we can create and query a TDFunc."""
    # Fail without enough arguments (only a is specified).
    tdf1 = TDFunc(_test_func, a=1.0)
    with pytest.raises(TypeError):
        _ = tdf1()

    # We succeed with a manually specified parameter (but first fail with
    # the default).
    tdf2 = TDFunc(_test_func, a=1.0, b=None)
    with pytest.raises(TypeError):
        _ = tdf2()
    assert tdf2(b=2.0) == 3.0

    # We can overwrite parameters.
    assert tdf2(a=3.0, b=2.0) == 5.0

    # Test that we ignore extra kwargs.
    assert tdf2(b=2.0, c=10.0, d=11.0) == 3.0

    # That we can use a different ordering for parameters.
    tdf3 = TDFunc(_test_func, b=2.0, a=1.0)
    assert tdf3() == 3.0


def test_np_sampler_method():
    """Test that we can wrap numpy random functions."""
    rng = np.random.default_rng(1001)
    tdf = TDFunc(rng.normal, loc=10.0, scale=1.0)

    # Sample 1000 times with the default values. Check that we are near
    # the expected mean and not everything is equal.
    vals = np.array([tdf() for _ in range(1000)])
    assert abs(np.mean(vals) - 10.0) < 1.0
    assert not np.all(vals == vals[0])

    # Override the mean and resample.
    vals = np.array([tdf(loc=25.0) for _ in range(1000)])
    assert abs(np.mean(vals) - 25.0) < 1.0
    assert not np.all(vals == vals[0])


def test_tdfunc_obj():
    """Test that we can create and query a TDFunc that depends on an object."""
    model = _StaticModel(a=10.0, b=11.0)

    # Check a function without defaults.
    tdf1 = TDFunc(_test_func, object_args=["a", "b"])
    assert model.eval_func(tdf1) == 21.0

    # Defaults set are overwritten by the object.
    tdf2 = TDFunc(_test_func, object_args=["a", "b"], a=1.0, b=0.0)
    assert model.eval_func(tdf2) == 21.0

    # But we can overwrite everything with kwargs.
    assert model.eval_func(tdf2, b=7.5) == 17.5
