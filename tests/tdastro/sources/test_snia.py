import numpy as np
from astropy import units as u
from tdastro.astro_utils.opsim import (
    get_pointings_matched_times,
    load_opsim_table,
    pointings_from_opsim,
)
from tdastro.astro_utils.passbands import Passband, PassbandGroup
from tdastro.astro_utils.snia_utils import DistModFromRedshift, HostmassX1Func, X0FromDistMod
from tdastro.effects.redshift import Redshift
from tdastro.sources.sncomso_models import SncosmoWrapperModel
from tdastro.sources.snia_host import SNIaHost
from tdastro.util_nodes.np_random import NumpyRandomFunc
from tdastro.astro_utils.unit_utils import flam_to_fnu

def test_snia_end2end(opsim=False,nsample=10):
    """Test that we can sample and create a salt3 object."""
    # Create a host galaxy anywhere on the sky.
    host = SNIaHost(
        ra=NumpyRandomFunc("uniform", low=0.0, high=360.0),
        dec=NumpyRandomFunc("uniform", low=-90.0, high=90.0),
        hostmass=NumpyRandomFunc("uniform", low=7, high=12),
        redshift=NumpyRandomFunc("uniform", low=0.01, high=0.02),
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
        t0=NumpyRandomFunc("uniform", low=60796, high=64448),
        x0=x0_func,
        x1=x1_func,
        c=c_func,
        ra=NumpyRandomFunc("normal", loc=host.ra, scale=0.01),
        dec=NumpyRandomFunc("normal", loc=host.dec, scale=0.01),
        redshift=host.redshift,
    )

    source.add_effect(Redshift(redshift=source.redshift, t0=source.t0))

    phase_rest = np.linspace(-5, 30, 20)
    wavelengths_rest = np.linspace(2000, 11000, 200)

    res = {
        "wavelengths_rest": wavelengths_rest,
        "phase_rest": phase_rest,
        "flux_flam": [],
        "flux_fnu": [],
        "parameter_values": [],
        "bandfluxes": [],
        "times": [],
    }

    passbands = PassbandGroup(
        passbands=[
            Passband(
                "LSST", "g", table_url="https://github.com/lsst/throughputs/blob/main/baseline/total_g.dat"
            ),
            Passband(
                "LSST", "r", table_url="https://github.com/lsst/throughputs/blob/main/baseline/total_r.dat"
            ),
        ]
    )

    if opsim:
        opsim_file = "/Users/mi/Work/tdastro/opsim_db/baseline_v3.4_10yrs.db"
        opsim_table = load_opsim_table(opsim_file)

        pointings = pointings_from_opsim(opsim_table)

    for _n in range(0, nsample):
        state = source.sample_parameters()

        z = source.get_param(state, "redshift")
        wave_obs = wavelengths_rest * (1.0 + z)
        phase_obs = phase_rest * (1.0 + z)

        if opsim:
            ra = source.get_param(state, "ra")
            dec = source.get_param(state, "dec")
            times = get_pointings_matched_times(pointings, ra, dec, fov=1.75)
        # Note that we don't have filter info yet.

        t0 = source.get_param(state, "t0")

        times = t0 + phase_obs

        res["times"].append(times)

        p = {}
        for parname in ["t0", "x0", "x1", "c", "redshift"]:
            p[parname] = source.get_param(state, parname)
        for parname in ["hostmass"]:
            p[parname] = host.get_param(state, parname)
        for parname in ["distmod"]:
            p[parname] = x0_func.get_param(state, parname)

        res["parameter_values"].append(p)

        flux_flam = source.evaluate(times, wave_obs, graph_state=state)
        res["flux_flam"].append(flux_flam)

        # convert ergs/s/cm^2/A to nJy

        flux_fnu = flam_to_fnu(flux_flam, wave_obs, wave_unit=u.AA, flam_unit= u.erg / u.second / u.cm**2 / u.AA, fnu_unit=u.nJy)

        res["flux_fnu"].append(flux_fnu)

        bandfluxes = passbands.fluxes_to_bandfluxes(flux_fnu, wave_obs)
        res["bandfluxes"].append(bandfluxes)

    return res





