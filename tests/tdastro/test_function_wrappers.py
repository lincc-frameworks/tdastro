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
    tdf1 = TDFunc(_test_func, a=1.0)

    # Fail without enough arguments (only a is specified).
    with pytest.raises(TypeError):
        _ = tdf1()

    # We succeed with a manually specified parameter.
    assert tdf1(b=2.0) == 3.0

    # We can overwrite parameters.
    assert tdf1(a=3.0, b=2.0) == 5.0


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
