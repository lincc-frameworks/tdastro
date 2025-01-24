"""Utility functions for AGN models.function

Adapted from https://github.com/RickKessler/SNANA/blob/master/src/gensed_AGN.py with authors' permission.
"""

from tdastro.consts import M_SUN_G


def agn_accretion_rate(blackhole_mass):
    """Compute the accretion rate at Eddington luminosity.

    Parameters
    ----------
    blackhole_mass : float
        The black hole mass in g.

    Returns
    -------
    accretion_rate : float
        The accretion rate at Eddington luminosity in g/s.
    """
    accretion_rate = 1.4e18 * blackhole_mass / M_SUN_G
    return accretion_rate


def agn_bolometric_luminosity(edd_ratio, blackhole_mass):
    """Compute the bolometric luminosity of an AGN.

    Parameters
    ----------
    edd_ratio : float
        The Eddington ratio.
    blackhole_mass : float
        The black hole mass in g.

    Returns
    -------
    L_bol : float
        The bolometric luminosity in erg/s.
    """
    L_bol = edd_ratio * 1.26e38 * blackhole_mass / M_SUN_G
    return L_bol
