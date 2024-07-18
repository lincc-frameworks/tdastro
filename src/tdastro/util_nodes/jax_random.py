"""Wrapper classes for calling JAX random number generators."""

import jax.random

from tdastro.base_models import FunctionNode


class JaxRandomFunc(FunctionNode):
    """The base class for JAX random number generators.

    Attributes
    ----------
    _key : `jax._src.prng.PRNGKeyArray`

    Note
    ----
    Automatically splits keys each time ``compute()`` is called, so
    each call produces a new pseudorandom number.
    """

    def __init__(self, func, **kwargs):
        super().__init__(func, **kwargs)

    def _update_object_seed(self, new_value):
        """Update the object seed to the new value.

        Parameters
        ----------
        new_value : `int`
             The new seed value.
        """
        self._object_seed = new_value
        self._key = jax.random.key(self._object_seed)

    def compute(self, **kwargs):
        """Execute the wrapped JAX sampling function.

        Parameters
        ----------
        **kwargs : `dict`, optional
            Additional function arguments.

        Raises
        ------
        ``ValueError`` is ``func`` attribute is ``None``.
        """
        if self.func is None:
            raise ValueError(
                "func parameter is None for a JAXRandom. You need to either "
                "set func or override compute()."
            )

        args = self._build_args_dict(**kwargs)
        self._key, subkey = jax.random.split(self._key)
        return float(self.func(subkey, **args))


class JaxRandomNormal(JaxRandomFunc):
    """A wrapper for the JAX normal function that takes
    a mean and std.

    Attributes
    ----------
    loc : `float`
        The mean of the distribution.
    scale : `float`
        The std of the distribution.
    """

    def __init__(self, loc, scale, **kwargs):
        super().__init__(jax.random.normal, **kwargs)

        # The mean and std as attributes, but not arguments.
        self.add_parameter("loc", loc)
        self.add_parameter("scale", scale)

    def compute(self, **kwargs):
        """Generate a random number from a normal distribution
        with the given mean and std.

        Parameters
        ----------
        **kwargs : `dict`, optional
            Additional function arguments.
        """
        initial_value = super().compute(**kwargs)
        local_mean = kwargs.get("loc", self.loc)
        local_std = kwargs.get("scale", self.scale)
        return local_std * initial_value + local_mean
