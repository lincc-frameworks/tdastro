import logging
import os
import socket
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal, Optional
from urllib.error import HTTPError, URLError

import numpy as np
import scipy.integrate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class PassbandGroup:
    """A group of passbands.

    Attributes
    ----------
    passbands : dict
        A dictionary of Passband objects.
    """

    def __init__(self, preset=None, passbands=None):
        """Initialize a PassbandGroup object.

        Attributes
        ----------
        preset : str, optional
            A pre-defined set of passbands to load.
        passbands : dict
            A dictionary of Passband objects assigned to the group. The keys are the passbands' full names:
            eg, LSST_u, LSST_g, HSC_u, etc.

        Parameters
        ----------
        preset : str, optional
            A pre-defined set of passbands to load.
        passbands : list, optional
            A list of Passband objects assigned to the group.
        """
        self.passbands = {}

        if preset is not None:
            self._load_preset(preset)

        if passbands is not None:
            for passband in passbands:
                self.passbands[passband.full_name] = passband  # This overrides any preset passbands

    def __str__(self) -> str:
        """Return a string representation of the PassbandGroup."""
        return (
            f"PassbandGroup containing {len(self.passbands)} passbands: "
            f"{', '.join(self.passbands.keys())}"
        )

    def _load_preset(self, preset):
        """Load a pre-defined set of passbands.

        Parameters
        ----------
        preset : str
            The name of the pre-defined set of passbands to load.
        """
        if preset == "LSST":
            self.passbands = {
                "LSST_u": Passband("LSST", "u"),
                "LSST_g": Passband("LSST", "g"),
                "LSST_r": Passband("LSST", "r"),
                "LSST_i": Passband("LSST", "i"),
                "LSST_z": Passband("LSST", "z"),
                "LSST_y": Passband("LSST", "y"),
            }
        else:
            raise ValueError(f"Unknown passband preset: {preset}")

    def fluxes_to_bandfluxes(self, flux_density_matrix, wavelengths_angstrom) -> np.ndarray:
        """Calculate bandfluxes for all passbands in the group.

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D array of flux densities where rows are times and columns are wavelengths.
        wavelengths_angstrom : np.ndarray
            An array of wavelengths in Angstroms corresponding to the columns of flux_density_matrix.

        Returns
        -------
        dict
            A dictionary of bandfluxes with passband labels as keys.
        """
        bandfluxes = {}
        for label, passband in self.passbands.items():
            bandfluxes[label] = passband.fluxes_to_bandflux(flux_density_matrix, wavelengths_angstrom)
        return bandfluxes


class Passband:
    """A passband contains information about its transmission curve and calculates its normalization.

    Attributes
    ----------
    label : str
        The label of the passband. Note that this is not unique across surveys.
    survey : str
        The survey to which the passband belongs.
    full_name : str
        The full name of the passband.
    table_path : str
        The path to the transmission table file.
    table_url : str
        The URL to download the transmission table file.
    transmission_table : np.ndarray
        A 2D array where the first col is wavelengths (Angstrom) and the second col is transmission values.
    normalized_transmission_table : np.ndarray
        A 2D array of wavelengths and normalized transmissions.
    """

    def __init__(
        self,
        survey: str,
        label: str,
        table_path: Optional[str] = None,
        table_url: Optional[str] = None,
        units: Optional[Literal["nm", "A"]] = "A",
    ):
        """Construct a Passband object

        Parameters
        ----------
        survey : str
            Which survey does this passband correspond to. Often "LSST"
        label : str
            Which label does this passband correspond to, Sometimes a filter name like "g"
        table_path : Optional[str], optional
            Path to the table defining the passband on the filesystem. Will take precedence over table_url
        table_url : Optional[str], optional
            URL to download the table from.
        units : Optional[Literal[&#39;nm&#39;,&#39;A&#39;]], optional
            Denotes whether the wavelength units of the table are nanometers ('nm') or Angstroms ('A').
            By default 'A'. Does not affect the output units of the class, only the interpretation of the
            provided passband table.
        """
        self.label = label
        self.survey = survey
        self.full_name = f"{survey}_{label}"
        self.transmission_table = self._load_transmission_table(
            table_path=table_path, table_url=table_url, units=units
        )
        self.normalized_transmission_table = self._normalize_transmission_table()

    def _load_transmission_table(
        self, table_path=None, table_url=None, units: Optional[Literal["nm", "A"]] = "A"
    ) -> np.ndarray:
        """Load a transmission table from a file or URL.

        Parameters
        ----------
        table_path : str, optional
            The path to the transmission table file. If None, a passbands directory will be created in the
            current directory.
        table_url : str, optional
            The URL to download the transmission table file, if it does not exist locally. If None, the URL
            will be constructed based on the survey and label. This is only available for the LSST survey at
            the moment.
        units : Optional[Literal[&#39;nm&#39;,&#39;A&#39;]], optional
            Denotes whether the wavelength units of the table are nanometers ('nm') or Angstroms ('A').
            By default 'A'. Does not affect the output units of the class, only the interpretation of the
            provided passband table.

        Returns
        -------
        np.ndarray
            A 2D array where the first column is wavelengths (Angstrom) and the second column is transmission
            values.
        """
        if table_path is None:
            if table_url is None:
                if self.survey == "LSST":
                    # TODO: This area will be changed with incoming Pooch data manager PR :)
                    # Consider: https://raw.githubusercontent.com/lsst/throughputs/e70d1daf069e606caa3feb43eccc62ec21e0baf5/baseline/total_g.dat
                    table_url = (
                        f"http://svo2.cab.inta-csic.es/svo/theory/fps3/getdata.php"
                        f"?format=ascii&id=LSST/LSST.{self.label}"
                    )
                else:
                    raise NotImplementedError(
                        f"Transmission table download is not yet implemented for survey: {self.survey}."
                        "Please provide a table URL."
                    )

            table_cache_dir = Path(__file__).parent / "passbands" / self.survey
            table_cache_dir.mkdir(exist_ok=True, parents=True)

            # Cache using the URL provided so when the URL changes we re-download
            # Primarily useful for local development
            table_cache_filename = f"{urllib.parse.quote_plus(table_url)}_{self.label}.dat"
            table_path = table_cache_dir / table_cache_filename

        if not table_path.exists():
            self._download_transmission_table(table_path, table_url)

        np_table = np.loadtxt(table_path)
        if units == "nm":
            np_table[:, 0] *= 10.0  # Multiply the first column (wavelength) by 10.0 to convert to Angstroms

        # Ensure the table is sorted by wavelength, which is the first column. Most data files do this
        # anyway, but we depend on a descending order in fluxes_to_bandpass()
        return np_table[np_table[:, 0].argsort()]

    def _download_transmission_table(self, table_path: Path, table_url: str) -> bool:
        """Download a transmission table from a URL.

        Parameters
        ----------
        table_path : str
            The path to save the downloaded transmission table.
        table_url : str
            The URL to download the transmission table file.

        Returns
        -------
        bool
            True if the download was successful, False otherwise.
        """
        try:
            socket.setdefaulttimeout(10)
            print(f"Retrieving {table_url}", flush=True)
            urllib.request.urlretrieve(table_url, table_path)
            if os.path.getsize(table_path) == 0:
                logging.error(f"Transmission table downloaded for {self.full_name} is empty.")
                return False
            else:
                logging.info(f"Downloaded: {self.full_name} transmission table.")
                return True
        except HTTPError as e:
            logging.error(f"HTTP error occurred when downloading table for {self.full_name}: {e}")
            return False
        except URLError as e:
            logging.error(f"URL error occurred when downloading table for {self.full_name}: {e}")
            return False

    def _normalize_transmission_table(self) -> np.ndarray:
        """Calculate the value of phi_b for all wavelengths in a transmission table.

        This is eq. 8 from "On the Choice of LSST Flux Units" (Ivezić et al.):

        φ_b(λ) = S_b(λ)λ⁻¹ / ∫ S_b(λ)λ⁻¹ dλ

        where S_b(λ) is the system response of the passband. Note that we use the transmission table
        here as our S_b(λ).

        Returns
        -------
        np.ndarray
            A 2D array of wavelengths and normalized transmissions.
        """
        wavelengths_angstrom = self.transmission_table[:, 0]
        transmissions = self.transmission_table[:, 1]
        # Calculate the numerators and denominator
        numerators = transmissions / wavelengths_angstrom
        denominator = scipy.integrate.trapezoid(numerators, x=wavelengths_angstrom)
        # Calculate phi_b for each wavelength
        normalized_transmissions = numerators / denominator
        return np.column_stack((wavelengths_angstrom, normalized_transmissions))

    def fluxes_to_bandflux(self, flux_density_matrix, wavelengths_angstrom) -> np.ndarray:
        """Calculate the bandflux for a given set of flux densities.

        This is eq. 7 from "On the Choice of LSST Flux Units" (Ivezić et al.):

        F_b = ∫ f(λ)φ_b(λ) dλ

        where f(λ) is the specific flux of an object at the top of the atmosphere, and φ_b(λ) is the
        normalized system response for given band b."

        Parameters
        ----------
        flux_density_matrix : np.ndarray
            A 2D array of flux densities where rows are times and columns are wavelengths.
        wavelengths_angstrom : np.ndarray
            An array of wavelengths in Angstroms corresponding to the columns of flux_density_matrix.

        Returns
        -------
        np.ndarray
            An array of bandfluxes with length flux_density_matrix, where each element is the bandflux
            at the corresponding time.
        """
        # TODO: This area will be changed with incoming interpolation/exterpolation PR :)
        # Check that the grid of wavelengths_angstrom matches the band's transmission table
        # TODO
        # For now, min exterpolation check: the bounds of the flux density grid are within the transmission
        # table bounds
        if (
            wavelengths_angstrom[-1] > self.normalized_transmission_table[-1, 0]
            or wavelengths_angstrom[0] < self.normalized_transmission_table[0, 0]
        ):
            raise ValueError(
                f"Extrapolation needed: flux density interval [{wavelengths_angstrom[0]}, "
                f"{wavelengths_angstrom[-1]}] is outside the transmission table interval "
                f"[{self.normalized_transmission_table[0, 0]}, {self.normalized_transmission_table[-1, 0]}]."
            )

        # Get only the flux densities that are within the passband's wavelength range
        passband_wavelengths = self.normalized_transmission_table[:, 0]
        passband_wavelengths_indices = np.searchsorted(wavelengths_angstrom, passband_wavelengths)

        # Since the passband wavelength table should (for the moment) always cover a wider range than the
        # wavelengths we are sampling, we will always find some passband wavelengths that index past the
        # end of of the wavelengths array. We truncate these past-the-end indiies and then pad the resulting
        # flux array out to the size of our passband table
        trunc_condition = passband_wavelengths_indices < len(wavelengths_angstrom)
        passband_wavelengths_indices_trunc = np.extract(trunc_condition, passband_wavelengths_indices)
        values_truncated = len(passband_wavelengths_indices) - len(passband_wavelengths_indices_trunc)

        # Create and pad flux density array
        passband_flux_density_matrix = flux_density_matrix[:, passband_wavelengths_indices_trunc]
        passband_flux_density_matrix = np.pad(
            passband_flux_density_matrix, [(0, 0), (0, values_truncated)], "constant", constant_values=0.0
        )

        # Calculate the bandflux as ∫ f(λ)φ_b(λ) dλ,
        # where f(λ) is the in-band flux density and φ_b(λ) is the normalized system response
        integrand = passband_flux_density_matrix * self.normalized_transmission_table[:, 1]
        bandfluxes = scipy.integrate.trapezoid(integrand, x=passband_wavelengths)

        return bandfluxes
