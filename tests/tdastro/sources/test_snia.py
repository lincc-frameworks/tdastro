import numpy as np
from tdastro.astro_utils.passbands import Passbands
from tdastro.astro_utils.snia_utils import DistModFromRedshift, HostmassX1Func, X0FromDistMod
from tdastro.effects.redshift import Redshift
from tdastro.sources.sncomso_models import SncosmoWrapperModel
from tdastro.sources.snia_host import SNIaHost
from tdastro.util_nodes.np_random import NumpyRandomFunc
from tdastro.astro_utils.opsim import (
    get_pointings_matched_times,
    load_opsim_table,
    pointings_from_opsim,
)


def test_snia():
    """Test that we can sample and create a salt3 object."""
    # Create a host galaxy anywhere on the sky.
    host = SNIaHost(
        ra=NumpyRandomFunc("uniform", low=0.0, high=360.0),
        dec=NumpyRandomFunc("uniform", low=-90.0, high=90.0),
        hostmass=NumpyRandomFunc("uniform", low=7, high=12),
        redshift=NumpyRandomFunc("uniform", low=0.01, high=1.0),
    )

    distmod_func = DistModFromRedshift(host.redshift, H0=73.0, Omega_m=0.3)
    x1_func = HostmassX1Func(host.hostmass)
    c_func = NumpyRandomFunc("normal", loc=0, scale=0.02)
    m_abs_func = NumpyRandomFunc("normal", loc=-19.3, scale=0.1)

    x0_func = X0FromDistMod(
        distmod=distmod_func,
        x1=x1_func,
        c=c_func,
        alpha=0.14,
        beta=3.1,
        m_abs=m_abs_func,
    )

    source = SncosmoWrapperModel(
        "salt2-h17",
        t0=0,
        x0=x0_func,
        x1=x1_func,
        c=c_func,
        ra=NumpyRandomFunc("normal", loc=host.ra, scale=0.01),
        dec=NumpyRandomFunc("normal", loc=host.dec, scale=0.01),
        redshift=host.redshift,
    )

    source.add_effect(Redshift(redshift=source.redshift, t0=source.t0))

    phase = np.linspace(-5, 30, 20)
    # times = phase
    wavelengths = np.linspace(3000, 8000, 200)
    # print(phase)
    # print(wavelengths)
    # print("parameters (x0,x1,c,hostmass,ra,dec,redshift):")

    res = {"wavelengths": wavelengths, "phase": phase, "flux": [], "parameter_values": [], "bandfluxes": []}

    passbands = Passbands(["g", "r", "i", "z"])

    passbands.load_all_transmission_tables()
    passbands.calculate_normalized_system_response_tables()

    # opsim_file = "/Users/mi/Work/tdastro/opsim_db/baseline_v3.4_10yrs.db"
    # opsim_table = load_opsim_table(opsim_file)

    # pointings = pointings_from_opsim(opsim_table)

    for _n in range(0, 10):
        flux = source.evaluate(phase, wavelengths, resample_parameters=True)
        p = source.get_all_parameter_values(True)

        # ra = p[]
        # dec = p[]
        # times = get_pointings_matched_times(pointings, ra, dec, fov=3.5)

        res["parameter_values"].append(p)
        res["flux"].append(flux)

        bandfluxes = passbands.get_all_in_band_fluxes(source, phase)
        res["bandfluxes"].append(bandfluxes)

    return res
