"""Utility functions for AGN models.function

Adapted from https://github.com/RickKessler/SNANA/blob/master/src/gensed_AGN.py with authors' permission.
"""

import numpy as np

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
        The accretion rate (ME_dot) at Eddington luminosity in g/s.
    """
    accretion_rate = 1.4e18 * blackhole_mass / M_SUN_G
    return accretion_rate


def agn_blackhole_accretion_rate(accretion_rate, edd_ratio):
    """Compute the accretion rate of the blackhole.

    Parameters
    ----------
    accretion_rate : float
        The accretion rate (ME_dot) at Eddington luminosity in g/s.
    edd_ratio : float
        The Eddington ratio.

    Returns
    -------
    bh_accretion_rate : float
        The accretion rate of the black hole in g/s.
    """
    bh_accretion_rate = accretion_rate * edd_ratio
    return bh_accretion_rate


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


def eddington_ratio_dist_fun(edd_ratio, galaxy_type="Blue", rng=None, num_samples=1):
    """Sample from the Eddington Ratio Distribution Function for a given galaxy type.

    Based on notebook from: https://github.com/burke86/imbh_forecast/blob/master/var.ipynb
    and the paper: https://ui.adsabs.harvard.edu/abs/2019ApJ...883..139S/abstract
    with parameters selected from: https://iopscience.iop.org/article/10.3847/1538-4357/aa803b/pdf

    Parameters
    ----------
    edd_ratio : float
        The Eddington ratio.
    galaxy_type : str, optional
        The type of galaxy, either 'Red' or 'Blue'.
        Default: 'Blue'
    rng : np.random.Generator, optional
        The random number generator.
        Default: None
    num_samples : int, optional
        The number of samples to draw. If 1 returns a float otherwise returns an array.
        Default: 1

    Returns
    -------
    result : float or np.array
        The Eddington ratio distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    if galaxy_type.lower() == "red":
        xi = 10**-2.13
        lambda_br = 10 ** rng.normal(-2.81, np.mean([0.22, 0.14]), num_samples)
        delta1 = rng.normal(0.41 - 0.7, np.mean([0.02, 0.02]), num_samples)  # > -0.45 won't affect LF
        delta2 = rng.normal(1.22, np.mean([0.19, 0.13]), num_samples)
    elif galaxy_type.lower() == "blue":
        xi = 10**-1.65
        lambda_br = 10 ** rng.normal(-1.84, np.mean([0.30, 0.37]), num_samples)
        delta1 = rng.normal(0.471 - 0.7, np.mean([0.02, 0.02]), num_samples)  # > -0.45 won't affect LF
        delta2 = rng.normal(2.53, np.mean([0.68, 0.38]), num_samples)
    else:
        raise ValueError("galaxy_type must be either 'Red' or 'Blue'.")

    result = xi * ((edd_ratio / lambda_br) ** delta1 + (edd_ratio / lambda_br) ** delta2)

    if num_samples == 1:
        return result[0]
    return result