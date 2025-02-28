import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from tdastro.astro_utils.noise_model import poisson_bandflux_std
from tdastro.consts import GAUSS_EFF_AREA2FWHM_SQ
from tdastro.opsim.opsim import OpSim

_ztf_opsim_colnames = {
    "maglim": "maglim",
    "sky": "scibckgnd",
    "fwhm": "fwhm",
    "dec": "dec",
    "exptime": "exptime",
    "filter": "filter",
    "ra": "ra",
    "time": "obsmjd",
    "zp": "zp",  # We add this column to the table
}
"""Default mapping of short column names to Rubin OpSim column names."""

ZTFCAM_PIXEL_SCALE = 1.01
"""The pixel scale for the ZTF camera in arcseconds per pixel."""

_ztfcam_readout_noise = 8
"""The standard deviation of the count of readout electrons per pixel for the ZTF camera."""

_ztfcam_dark_current = 0.0
"""The dark current for the ZTF camera in electrons per second per pixel."""

_ztfcam_view_radius = 2.735
"""The angular radius of the observation field (in degrees). ZTF FOV is 47 deg^2"""

_ztfcam_ccd_gain = 6.2
"""CCD gain (in e-/ADU)"""


def calculate_ztf_zero_points(
    maglim=None,
    sky=None,
    fwhm=None,
    gain=_ztfcam_ccd_gain,
    readnoise=_ztfcam_readout_noise,
    darkcurrent=_ztfcam_dark_current,
    exptime=None,
    nexposure=1,
):
    """
    Calculate zero points based on the 5-sigma mag limit.

    snr = flux/fluxerr
    fluxerr = sqrt(flux + sky*npix*gain + readnoise**2*nexposure*npix + darkcurrent*npix*exptime*nexposure)
    5 = flux/fluxerr
    25 = flux**2/(flux + sky*npix*Gain + readnoise**2*nexposure*npix + darkcurrent*npix*exptime*nexposure)
    flux**2 - 25*flux -25*( sky*npix*Gain
                            + readnoise**2*nexposure*npix
                            + darkcurrent*npix*exptime*nexposure)
                        = 0
    flux = 12.5 + 0.5*sqrt(625
                            + 100( sky*npix*Gain
                            + readnoise**2*nexposure*npix
                            + darkcurrent*npix*exptime*nexposure) )
    zp = 2.5log(flux) + maglim

    Parameters
    ----------
    maglim : float or ndarray
        Five-sigma magnitude limit.
    sky : float or ndarry
        Sky background in ADU/pixel.
    fwhm : float or ndarray
        PSF in pixels.
    gain : float or ndarray; default is _ztfcam_ccd_gain
        CCD gain.
    readnoise : float or ndarray; default is _ztfcam_readout_noise
        Read noise (in e-/pixel).
    darkcurrent : float or ndarray; default is _ztfcam_dark_current
        Dark current (in e-/pixel/second).
    exptime : float or ndarray
        Exposure time (in seconds).
    nexposure : int or ndarray
        Number of exposure.

    Returns
    -------
    zp: float or ndarray
        Instrument zero point.
    """

    npix = 2.266 * fwhm**2  # =4*pi*sigma**2=pi/2/ln2 * FWHM**2
    flux_at_5sigma_limit = 12.5 + 2.5 * np.sqrt(
        25.0
        + 4.0
        * (sky * npix * gain + readnoise**2 * nexposure * npix + darkcurrent * npix * exptime * nexposure)
    )
    zp = 2.5 * np.log10(flux_at_5sigma_limit / gain) + maglim

    return zp


class ZTFOpsim(OpSim):
    """A subclass for ZTF exposure table.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the OpSim information.
    colmap : dict
        A mapping of short column names to their names in the underlying table.
        Defaults to the Rubin OpSim column names, stored in _rubin_opsim_colnames:
        {_ztf_opsim_colnames}

    Attributes
    ----------
    table : pandas.core.frame.DataFrame
        The table with all the OpSim information.
    colmap : dict
        A mapping of short column names to their names in the underlying table.
    _kd_tree : scipy.spatial.KDTree or None
        A kd_tree of the OpSim pointings for fast spatial queries. We use the scipy
        kd-tree instead of astropy's functions so we can directly control caching.
    pixel_scale : float or None, optional
        The pixel scale for the ZTF camera in arcseconds per pixel.
    read_noise : float or None, optional
        The readout noise for the ZTF camera in electrons per pixel.
    dark_current : float or None, optional
        The dark current for the ZTF camera in electrons per second per pixel.
    radius : float
        The angular radius of the observations (in degrees).
    """

    def __init__(self, table, colmap=_ztf_opsim_colnames):
        if isinstance(table, dict):
            self.table = pd.DataFrame(table)
        else:
            self.table = table

        # replace invalid values in table
        self.table = self.table.replace("", np.nan)
        self.table = self.table.dropna(subset=["fwhm"])

        self.colmap = colmap
        self.radius = _ztfcam_view_radius
        self.pixel_scale = ZTFCAM_PIXEL_SCALE
        self.dark_current = _ztfcam_dark_current
        self.read_noise = _ztfcam_readout_noise
        self.gain = _ztfcam_ccd_gain

        # Build the kd-tree.
        self._kd_tree = None
        self._build_kd_tree()

        # If we are not given zero point data, try to derive it from the other columns.
        if not self.has_columns("zp"):
            if self.has_columns(["maglim", "sky", "fwhm", "exptime"]):
                self._assign_zero_points()
            else:
                raise ValueError(
                    "OpSim must include either a zero point column or the columns "
                    "needed to derive it (maglim, sky, fwhm and exptime)."
                )
        # Convert obsdate to mjd and add column
        obsdate = self.table[self.colmap.get("obsdate", "obsdate")].tolist()
        t = Time(obsdate, format="iso", scale="utc")
        mjd = t.mjd
        self.add_column(self.colmap.get("obsmjd", "obsmjd"), mjd, overwrite=True)

    def _assign_zero_points(self):
        """Assign instrumental zero points in ADU to the OpSim tables."""

        zp_values = calculate_ztf_zero_points(
            maglim=self.table[self.colmap.get("maglim", "maglim")],
            sky=self.table[self.colmap.get("sky", "sky")],
            fwhm=self.table[self.colmap.get("fwhm", "fwhm")],
            exptime=self.table[self.colmap.get("exptime", "exptime")],
        )
        self.add_column(self.colmap.get("zp", "zp"), zp_values, overwrite=True)

    @classmethod
    def from_db(cls, filename, sql_query="SELECT * from exposures", colmap=_ztf_opsim_colnames):
        """Create an OpSim object from the data in an opsim db file.

        Parameters
        ----------
        filename : str
            The name of the opsim db file.
        sql_query : str
            The SQL query to use when loading the table.
            Default = "SELECT * FROM observations"
        colmap : dict
            A mapping of short column names to their names in the underlying table.
            Defaults to the Rubin opsim column names.

        Returns
        -------
        opsim : OpSim
            A table with all of the pointing data.

        Raise
        -----
        FileNotFoundError if the file does not exist.
        ValueError if unable to load the table.
        """

        if not Path(filename).is_file():
            raise FileNotFoundError(f"opsim file {filename} not found.")
        con = sqlite3.connect(f"file:{filename}?mode=ro", uri=True)

        # Read the table.
        try:
            opsim = pd.read_sql_query(sql_query, con)
        except Exception:
            raise ValueError("Opsim database read failed.") from None

        # Close the connection.
        con.close()

        return ZTFOpsim(opsim, colmap=colmap)

    def bandflux_error_point_source(self, bandflux, index):
        """Compute observational bandflux error for a point source

        Parameters
        ----------
        bandflux : array_like of float
            Band bandflux of the point source in nJy.
        index : array_like of int
            The index of the observation in the OpSim table.

        Returns
        -------
        flux_err : array_like of float
            Simulated bandflux noise in nJy.
        """
        observations = self.table.iloc[index]

        # By the effective FWHM definition, see
        # https://smtn-002.lsst.io/v/OPSIM-1171/index.html
        footprint = GAUSS_EFF_AREA2FWHM_SQ * observations["fwhm"] ** 2  # in pixels

        return poisson_bandflux_std(
            bandflux,  # ADU
            pixel_scale=1.0,
            total_exposure_time=observations["exptime"],
            exposure_count=1,
            footprint=footprint,
            sky=observations["scibckgnd"],  # ADU/pixel
            zp=observations["zp"],  # ADU
            readout_noise=self.read_noise,  # e-/pixel
            dark_current=self.dark_current,  # e-/second/pixel
        )


def create_random_ztf_opsim(num_obs, seed=None):
    """Create a random OpSim pointings drawn uniformly from (RA, dec).

    Parameters
    ----------
    num_obs : int
        The size of the OpSim to generate.
    seed : int
        The seed to used for random number generation. If None then
        uses a default random number generator.
        Default: None

    Returns
    -------
    opsim_data : OpSim
        The OpSim data structure.
    seed : int, optional
        The seed for the random number generator.
    """
    if num_obs <= 0:
        raise ValueError("Number of observations must be greater than zero.")

    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed=seed)

    # Generate the (RA, dec) pairs uniformly on the surface of a sphere.
    ra = np.degrees(rng.uniform(0.0, 2.0 * np.pi, size=num_obs))
    dec = np.degrees(np.arccos(2.0 * rng.uniform(0.0, 1.0, size=num_obs) - 1.0) - (np.pi / 2.0))

    # Generate the information needed to compute zeropoint.
    maglim = rng.uniform(19.0, 21.0, size=num_obs)
    sky = rng.uniform(100.0, 200.0, size=num_obs)
    fwhm = rng.uniform(1.0, 3.0, size=num_obs)
    filter = rng.choice(["g", "r", "i"], size=num_obs)

    input_data = {
        "obsdate": ["2018-03-25 06:04:40.000"] * num_obs,
        "ra": ra,
        "dec": dec,
        "maglim": maglim,
        "scibckgnd": sky,
        "fwhm": fwhm,
        "filter": filter,
        "exptime": 30.0 * np.ones(num_obs),
    }

    opsim = ZTFOpsim(input_data, colmap=_ztf_opsim_colnames)

    return opsim
