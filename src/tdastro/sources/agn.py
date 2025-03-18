"""A model for an AGN.

Adapted from https://github.com/RickKessler/SNANA/blob/master/src/gensed_AGN.py
with the authors' permission.
"""

from os import urandom

import numpy as np
from astropy import constants
from citation_compass import cite_function
from scipy import integrate

from tdastro.base_models import FunctionNode
from tdastro.consts import M_SUN_G
from tdastro.sources.physical_model import PhysicalModel


def sample_damped_random_walk(times, tau_v, sf_inf, t0, rng=None):
    """Sample a damped random walk.

    Parameters
    ----------
    times : np.ndarray
        A length T array with the times at which to sample the damped random
        walk (in MJD).
    tau_v : np.ndarray
        A length W array with timescale of the damped random walk (in s).
    sf_inf : np.ndarray
        A length W array with structure function at infinity time in magnitude.
    t0 : float
        The initial time moment (in MJD).
    rng : np.random.Generator, optional
        The random number generator to use.
        Default: None

    Returns
    -------
    samples : float
        The sampled value of the damped random walk.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(sf_inf) != len(tau_v):
        raise ValueError(
            "The arrays sf_inf and tau_v must have the same length. "
            f"Received lengths: sf_inf={len(sf_inf)}, tau_v={len(tau_v)}"
        )
    samples = np.zeros((len(times), len(tau_v)))

    # Set the initial values.
    curr_t = t0
    delta_m = rng.random() * sf_inf

    # Iterate over each time step.
    for idx, time in enumerate(times):
        if time <= curr_t and idx != 0:
            raise ValueError("Times must be monotonically increasing.")

        dt = time - curr_t
        delta_m = delta_m * np.exp(-dt / tau_v) + sf_inf * np.sqrt(1 - np.exp(-2 * dt / tau_v)) * rng.random()
        samples[idx, :] = delta_m

    return samples


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
    seed : int, optional
        The seed to use for the random number generator.
        Default: None
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(self, t0, blackhole_mass, edd_ratio, seed=None, **kwargs):
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

        # Set the random number generator for this object.
        if seed is None:
            seed = int.from_bytes(urandom(4), "big")
        self._rng = np.random.default_rng(seed=seed)

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
    @cite_function
    def compute_flux_standard_disk(Mdot, nu, rin, i, d, M):
        """Compute the flux based on a standard disk model.

        References
        ----------
        Lipunova, G., Malanchev, K., Shakura, N. (2018)
        https://doi.org/10.1007/978-3-319-93009-1_1

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
    @cite_function
    def compute_mag_i(bolometric_luminosity):
        """Compute the i band magnitude from the bolometric luminosity.

        References
        ----------
        Shen et al., 2013 - https://adsabs.harvard.edu/full/2013BASI...41...61S

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
    @cite_function
    def compute_r_0(r_in):
        """Compute the initial radius of the ring (r_0) in a standard disk model
        given the inner radius.

        References
        ----------
        Lipunova, G., Malanchev, K., Shakura, N. (2018)
        https://doi.org/10.1007/978-3-319-93009-1_1

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
    @cite_function
    def compute_structure_function_at_inf(wavelength, mag_i=-23, blackhole_mass=1e9 * M_SUN_G):
        """Compute the structure function at infinity time in magnitude.

        References
        ----------
        Suberlak et al. 2021 - DOI 10.3847/1538-4357/abc698

        Parameters
        ----------
        wavelength : np.ndarray
            A length W array with the frequencies in Hz.
        mag_i : float, optional
            The i band magnitude.
            Default: -23
        blackhole_mass : float, optional
            The black hole mass in g.
            Default: 1e9 * M_SUN_G

        Returns
        -------
        result : np.ndarray
            A length W array with structure function at infinity time in magnitude.
        """
        # Equation and parameters for A=-0.51, B=-0.479, C=0.13, and D=0.18
        #  adopted from Suberlak et al. 2021: DOI 10.3847/1538-4357/abc698
        return 10 ** (
            -0.51
            - 0.479 * np.log10(wavelength / (4000e-8))
            + 0.13 * (mag_i + 23)
            + 0.18 * np.log10(blackhole_mass / (1e9 * M_SUN_G))
        )

    @staticmethod
    @cite_function
    def compute_tau_v_drw(wavelength, mag_i=-23, blackhole_mass=1e9 * M_SUN_G):
        """Compute the timescale (tau_v) for the DRW model.

        References
        ----------
        Suberlak et al. 2021 - DOI 10.3847/1538-4357/abc698

        Parameters
        ----------
        wavelength : np.ndarray
            A length W array with the frequenc in Hz.
        mag_i : float, optional
            The i band magnitude.
            Default: -23
        blackhole_mass : float, optional
            The black hole mass in g.
            Default: 1e9 * M_SUN_G

        Returns
        -------
        tau_v : np.ndarray
            A length W array with the timescale in s for each wavelength.
        """
        # Equation and parameters for A=2.4, B=0.17, C=0.03, and D=0.21 adopted
        # from Suberlak et al. 2021: DOI 10.3847/1538-4357/abc698
        return 10 ** (
            2.4
            + 0.17 * np.log10(wavelength / (4000e-8))
            + 0.03 * (mag_i + 23)
            + 0.21 * np.log10(blackhole_mass / (1e9 * M_SUN_G))
        )

    @staticmethod
    @cite_function
    def compute_temp_at_r_0(M, Mdot, r_in):
        """Compute the effective temperature at r0. This is the same as the maximum effective
        temperature at the disc surface (Tmax).

        References
        ----------
        Lipunova, G., Malanchev, K., Shakura, N. (2018)
        https://doi.org/10.1007/978-3-319-93009-1_1

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

    def compute_flux(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of rest frame timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths (in angstroms).
        graph_state : `GraphState`
            An object mapping graph parameters to their values.
        **kwargs : `dict`, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values (in nJy).
        """
        params = self.get_local_params(graph_state)

        # Compute the parameters for these wavelengths.
        tau_v = self.compute_tau_v_drw(wavelengths, params["mag_i"], params["blackhole_mass"])
        sf_inf = self.compute_structure_function_at_inf(
            wavelengths, params["mag_i"], params["blackhole_mass"]
        )

        # Compute the average flux of a standard disk model. Use a factor of 2 (two sides
        # of the disk) to get the total flux.
        fnu_average = 2.0 * self.compute_flux_standard_disk(
            params["blackhole_accretion_rate"],
            constants.c.cgs.value / wavelengths,  # nu
            1,  # rin
            0,  # i
            1.0,  # d
            params["blackhole_mass"],
        )

        # Run the damped random walk.
        delta_m = sample_damped_random_walk(
            times,
            tau_v,
            sf_inf,
            params["t0"],
            rng=self._rng,
        )

        flux_density = 10 ** (-0.4 * delta_m) * fnu_average
        return flux_density
