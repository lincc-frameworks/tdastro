import astropy.cosmology.units as cu
from astropy import units as u

from tdastro.function_wrappers import TDFunc


def redshift_to_distance(redshift, cosmology, kind="comoving"):
    """Compute a source's distance given its redshift and a
    specified cosmology using astropy's redshift_distance().

    Parameters
    ----------
    redshift : `float`
        The redshift value.
    cosmology : `astropy.cosmology`
        The cosmology specification.
    kind : `str`
        The distance type for the Equivalency as defined by
        astropy.cosmology.units.redshift_distance.

    Returns
    -------
    distance : `float`
        The distance (in pc)
    """
    z = redshift * cu.redshift
    distance = z.to(u.pc, cu.redshift_distance(cosmology, kind=kind))
    return distance.value


class RedshiftDistFunc(TDFunc):
    """A wrapper class for the redshift_to_distance() function.

    Attributes
    ----------
    cosmology : `astropy.cosmology`
        The cosmology specification.
    kind : `str`
        The distance type for the Equivalency as defined by
        astropy.cosmology.units.redshift_distance.
    """

    def __init__(self, cosmology, kind="comoving", **kwargs):
        self.cosmology = cosmology
        self.kind = kind
        default_args = {"cosmology": cosmology, "kind": kind}

        # Call the super class's constructor with the needed information.
        # Do not pass kwargs because we are limiting the arguments to match
        # the signature of redshift_to_distance().
        super().__init__(
            func=redshift_to_distance,
            default_args=default_args,
            object_args=["redshift"],
        )

    def __str__(self):
        """Return the string representation of the function."""
        return f"RedshiftDistFunc({self.cosmology.name}, {self.kind})"
