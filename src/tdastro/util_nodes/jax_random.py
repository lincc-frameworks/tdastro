"Wrapper classes for calling JAX random number generators."

import jax.random

from tdastro.base_models import FunctionNode

class JAXRandom(FunctionNode):
    """The base class for JAX random number generators.
    
    Attributes
    ----------
    _key : `jax._src.prng.PRNGKeyArray`
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        


def redshift_to_distance(redshift, cosmology):
    """Compute a source's luminosity distance given its redshift and a
    specified cosmology using astropy's redshift_distance().

    Parameters
    ----------
    redshift : `float`
        The redshift value.
    cosmology : `astropy.cosmology`
        The cosmology specification.

    Returns
    -------
    distance : `float`
        The luminosity distance (in pc)
    """
    z = redshift * cu.redshift
    distance = z.to(u.pc, cu.redshift_distance(cosmology, kind="luminosity"))
    return distance.value


class RedshiftDistFunc(FunctionNode):
    """A wrapper class for the redshift_to_distance() function.

    Attributes
    ----------
    cosmology : `astropy.cosmology`
        The cosmology specification.
    kind : `str`
        The distance type for the Equivalency as defined by
        astropy.cosmology.units.redshift_distance.

    Parameters
    ----------
    redshift : function or constant
        The function or constant providing the redshift value.
    cosmology : `astropy.cosmology`
        The cosmology specification.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, redshift, cosmology, **kwargs):
        # Call the super class's constructor with the needed information.
        super().__init__(
            func=redshift_to_distance,
            redshift=redshift,
            cosmology=cosmology,
            **kwargs,
        )
