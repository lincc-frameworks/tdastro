"""A model for a gensed AGN.

Adapted from https://github.com/RickKessler/SNANA/blob/master/src/gensed_AGN.py
with the authors' permission.
"""

import numpy as np
from astropy import constants, units
from scipy import integrate

from tdastro.base_models import FunctionNode
from tdastro.consts import M_SUN_G
from tdastro.sources.physical_model import PhysicalModel


class AGN(PhysicalModel):
    """A model for an AGN.

    Parameterized values include:
      * accretion_rate - The accretion rate (ME_dot) at Eddington luminosity in g/s.
      * blackhole_accretion_rate - The accretion rate of the black hole in g/s.
      * blackhole_mass - The black hole mass in g.
      * edd_ratio - The Eddington ratio.
      * dec - The object's declination in degrees. [from PhysicalModel]
      * distance - The object's luminosity distance in pc. [from PhysicalModel]
      * L_bol - The bolometric luminosity in erg/s.
      * mag_i - The i band absolute magnitude.
      * ra - The object's right ascension in degrees. [from PhysicalModel]
      * redshift - The object's redshift. [from PhysicalModel]
      * t0 - The t0 of the zero phase, date. [from PhysicalModel]

    Parameters
    ----------
    t0 : float
        initial time moment in days.
    blackhole_mass : float
        The black hole mass in g.
    edd_ratio: float
        Eddington ratio
    """

    def __init__(self, t0, blackhole_mass, edd_ratio, **kwargs):
        super().__init__(t0=t0, **kwargs)

        # Add the parameters for the AGN. t0 already set in PhysicalModel.
        self.add_parameter("blackhole_mass", blackhole_mass, **kwargs)
        self.add_parameter("edd_ratio", edd_ratio, **kwargs)

        # Add the derived parameters using FunctionNodes built from the object's static methods.
        # Each of these will be computed for each sample value of the input parameters.
        self.add_parameter(
            "critical_accretion_rate",
            FunctionNode(self.compute_critical_accretion_rate, blackhole_mass=self.blackhole_mass),
            **kwargs,
        )
        self.add_parameter(
            "blackhole_accretion_rate",
            FunctionNode(
                self.compute_blackhole_accretion_rate,
                accretion_rate=self.critical_accretion_rate,  # Pull from computed accretion rate
                edd_ratio=self.edd_ratio,  # Pull from sampled ratio
            ),
            **kwargs,
        )
        self.add_parameter(
            "bolometric_luminosity",
            FunctionNode(
                self.compute_bolometric_luminosity,
                edd_ratio=self.edd_ratio,  # Pull from sampled ratio
                blackhole_mass=self.blackhole_mass,  # Pull from the sampled mass
            ),
            **kwargs,
        )
        self.add_parameter(
            "mag_i",
            FunctionNode(
                self.compute_mag_i,
                bolometric_luminosity=self.bolometric_luminosity,  # Pull from computed value.
            ),
            **kwargs,
        )

        # TODO: Figure out how to sample delta_m, fnu_avergae, sf_ind, and tau_v.

    # ------------------------------------------------------------------------
    # --- Static helper methods for computing the derived parameters. --------
    # ------------------------------------------------------------------------

    @staticmethod
    def compute_critical_accretion_rate(blackhole_mass):
        """Compute the critical accretion rate at Eddington luminosity.

        Parameters
        ----------
        blackhole_mass : float
            The black hole mass in g.

        Returns
        -------
        accretion_rate : float
            The accretion rate (ME_dot) at Eddington luminosity in g/s.
        """
        return 1.4e18 * blackhole_mass / M_SUN_G

    @staticmethod
    def compute_blackhole_accretion_rate(accretion_rate, edd_ratio):
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
        return accretion_rate * edd_ratio

    @staticmethod
    def compute_bolometric_luminosity(edd_ratio, blackhole_mass):
        """Compute the bolometric luminosity of an AGN.

        Parameters
        ----------
        edd_ratio : float
            The Eddington ratio.
        blackhole_mass : float
            The black hole mass in g.

        Returns
        -------
        bolometric_luminosity : float
            The bolometric luminosity in erg/s.
        """
        return edd_ratio * 1.26e38 * blackhole_mass / M_SUN_G

    @staticmethod
    def compute_mag_i(bolometric_luminosity):
        """Compute the i band magnitude from the bolometric luminosity.

        Parameters
        ----------
        bolometric_luminosity : float
            The bolometric luminosity in erg/s.

        Returns
        -------
        mag_i : float
            The i band magnitude.
        """
        # Adpated from Shen et al., 2013: https://adsabs.harvard.edu/full/2013BASI...41...61S
        return 90 - 2.5 * np.log10(bolometric_luminosity)

    @staticmethod
    def compute_r_0(r_in):
        """Compute the initial radius of the ring (r_0) in a standard disk model
        given the inner radius.

        Parameters
        ----------
        r_in : float
            The inner radius of the accretion disk.

        Returns
        -------
        r_0 : float
            The initial radius of the ring.
        """
        # Adapted from Lipunova, G., Malanchev, K., Shakura, N. (2018). Page 33 for the main equation
        # DOI https://doi.org/10.1007/978-3-319-93009-1_1
        return (7 / 6) ** 2 * r_in

    @staticmethod
    def compute_temp_at_r_0(M, Mdot, r_in):
        """Compute the effective temperature at r0. This is the same as the maximum effective
        temperature at the disc surface (Tmax).

        Parameters
        ----------
        M : float
            The mass of the gravitating centre.
        Mdot : float
            The accretion rate at the previous time step.
        r_in : float
            The inner radius of the accretion disk.

        Returns
        -------
        T_0 : float
            The effective temperature at r0.
        """
        # Lipunova, G., Malanchev, K., Shakura, N. (2018). Page 33 for the main equation
        # DOI https://doi.org/10.1007/978-3-319-93009-1_1
        sigma_sb = constants.sigma_sb.cgs.value
        G = constants.G.cgs.value
        return 2 ** (3 / 4) * (3 / 7) ** (7 / 4) * (G * M * Mdot / (np.pi * sigma_sb * r_in**3)) ** (1 / 4)

    @staticmethod
    def compute_x_fun(nu, T0, r, r0):
        """Compute the variable of integration x.

        Parameters
        ----------
        nu : float
            The frequency.
        T0 : float
            The effective temperature at r0.
        r : float
            The radius of the accretion disk.
        r0 : float
            The initial radius of the ring.

        Returns
        -------
        x : float
            The variable of integration x.
        """
        # Lipunova, G., Malanchev, K., Shakura, N. (2018). Page 33 for the main equation
        # DOI https://doi.org/10.1007/978-3-319-93009-1_1
        h = constants.h.cgs.value
        k_B = constants.k_B.cgs.value
        return h * nu / (k_B * T0) * (r / r0) ** (3 / 4)

    @staticmethod
    def flux_standard_disk(Mdot, nu, rin, i, d, M):
        """Compute the flux based on a standard disk model.

        Parameters
        ----------
        Mdot : float or np.ndarray
            The accretion rate at the previous time step.
        nu : float or np.ndarray
            The frequency.
        rin : float or np.ndarray
            The inner radius of the accretion disk.
        i : float or np.ndarray
            The inclination.
        d : float or np.ndarray
            The distance.
        M : float or np.ndarray
            The mass of the gravitating center.

        Returns
        -------
        flux : float
            The flux at the given time step.
        """
        # Compute the initial radius of the ring (r_0) and the effective temperature at r_0 (T_0).
        # Lipunova, G., Malanchev, K., Shakura, N. (2018). Page 33 for the main equation
        # DOI https://doi.org/10.1007/978-3-319-93009-1_1
        r_0 = AGN.compute_r_0(rin)
        T_0 = AGN.compute_temp_at_r_0(M, Mdot, rin)

        # large x in exponetial causes overflow, but 1/inf is zero.
        with np.errstate(over="ignore"):

            def _fun_integr(x):
                return (x ** (5 / 3)) / np.expm1(x)

            integ, _ = integrate.quad(_fun_integr, 1e-6, np.inf)

        h = constants.h.cgs.value
        k_B = constants.k_B.cgs.value
        c = constants.c.cgs.value
        return (
            (16 * np.pi)
            / (3 * d**2)
            * np.cos(i)
            * (k_B * T_0 / h) ** (8 / 3)
            * h
            * (nu ** (1 / 3))
            / (c**2)
            * (r_0**2)
            * integ
        )

    @staticmethod
    def twice_fnu_average_standard_disk(bh_accretion_rate, lam, blackhole_mass):
        """Compute twice the average flux of a standard disk model.

        Parameters
        ----------
        bh_accretion_rate : float
            The accretion rate of the black hole in g/s.
        lam : np.ndarray
            The wavelengths
        blackhole_mass : float
            The black hole mass in g.

        Returns
        -------
        flux : float
            The flux at the given time step.
        """
        flux_av = AGN.flux_standard_disk(
            bh_accretion_rate,
            constants.c.cgs.value / lam,
            rin=1,
            i=0,
            d=10 * (1 * units.pc).to_value(units.cm),
            M=blackhole_mass,
        )
        return 2.0 * flux_av

    @staticmethod
    def structure_function_at_inf(lam, mag_i=-23, blackhole_mass=1e9 * M_SUN_G):
        """Compute the structure function at infinity in magnitude.

        Parameters
        ----------
        lam : float
            The frequency in Hz.
        mag_i : float, optional
            The i band magnitude.
            Default: -23
        blackhole_mass : float, optional
            The black hole mass in g.
            Default: 1e9 * M_SUN_G

        Returns
        -------
        result : float
            The structure function at infinity in magnitude.
        """
        # Equation and parameters for A=-0.51, B=-0.479, C=0.13, and D=0.18
        #  adopted from Suberlak et al. 2021: DOI 10.3847/1538-4357/abc698
        return 10 ** (
            -0.51
            - 0.479 * np.log10(lam / (4000e-8))
            + 0.13 * (mag_i + 23)
            + 0.18 * np.log10(blackhole_mass / (1e9 * M_SUN_G))
        )

    @staticmethod
    def tau_v_drw(lam, mag_i=-23, blackhole_mass=1e9 * M_SUN_G):
        """Compute the timescale (tau_v) for the DRW model.

        Parameters
        ----------
        lam : np.ndarray
            The frequenc in Hz.
        mag_i : float, optional
            The i band magnitude.
            Default: -23
        blackhole_mass : float, optional
            The black hole mass in g.
            Default: 1e9 * M_SUN_G

        Returns
        -------
        tau_v : float
            The timescale in s.
        """
        # Equation and parameters for A=2.4, B=0.17, C=0.03, and D=0.21 adopted
        # from Suberlak et al. 2021: DOI 10.3847/1538-4357/abc698
        return 10 ** (
            2.4
            + 0.17 * np.log10(lam / (4000e-8))
            + 0.03 * (mag_i + 23)
            + 0.21 * np.log10(blackhole_mass / (1e9 * M_SUN_G))
        )

    @staticmethod
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
