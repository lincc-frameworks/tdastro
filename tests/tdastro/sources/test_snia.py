import numpy as np
import sncosmo
from astropy import units as u
from tdastro.astro_utils.opsim import OpSim
from tdastro.astro_utils.passbands import PassbandGroup
from tdastro.astro_utils.snia_utils import DistModFromRedshift, HostmassX1Func, X0FromDistMod
from tdastro.astro_utils.unit_utils import flam_to_fnu
from tdastro.rand_nodes.np_random import NumpyRandomFunc
from tdastro.sources.sncomso_models import SncosmoWrapperModel
from tdastro.sources.snia_host import SNIaHost


def draw_single_random_sn(
    source,
    wavelengths_rest=None,
    phase_rest=None,
    passbands=None,
    opsim=False,
    opsim_data=None,
    randseed=None,
):
    """
    Draw a single random SN realiztion
    """

    state = source.sample_parameters()

    z = source.get_param(state, "redshift")
    wave_obs = passbands.waves
    wavelengths_rest = wave_obs / (1.0 + z)
    phase_obs = phase_rest * (1.0 + z)

    res = {"wavelengths_rest": wavelengths_rest, "phase_rest": phase_rest}

    t0 = source.get_param(state, "t0")
    times = t0 + phase_obs

    if opsim:
        ra = source.get_param(state, "ra")
        dec = source.get_param(state, "dec")
        obs = opsim_data.get_observations(ra, dec, radius=1.75, cols=["time", "filter"])

        times = obs["time"]
        phase_obs = times - t0
        times = np.sort(times[(phase_obs > -20) & (phase_obs < 50)])
        # Note that we don't have filter info yet.

        if len(times) == 0:
            print(f"No overlap time in opsim for (ra,dec)=({ra:.2f},{dec:.2f})")
            return

    res["times"] = times

    flux_flam = source.evaluate(times, wave_obs, graph_state=state)
    res["flux_flam"] = flux_flam

    # convert ergs/s/cm^2/AA to nJy

    flux_fnu = flam_to_fnu(
        flux_flam, wave_obs, wave_unit=u.AA, flam_unit=u.erg / u.second / u.cm**2 / u.AA, fnu_unit=u.nJy
    )

    res["flux_fnu"] = flux_fnu

    bandfluxes = passbands.fluxes_to_bandfluxes(flux_fnu)
    res["bandfluxes"] = bandfluxes

    res["state"] = state

    return res


def test_snia_end2end(
    opsim_small,
    passbands_dir,
    opsim_db_file=None,
    opsim=True,
    nsample=1,
    return_result=False,
    phase_rest=None,
    wavelengths_rest=None,
):
    """Test that we can sample and create SN Ia simulation using the salt3 model."""

    opsim_data = OpSim.from_db(opsim_db_file if opsim_db_file else opsim_small) if opsim else None

    ra_min = opsim_data["fieldRA"].min() if opsim_data else 0.0
    ra_max = opsim_data["fieldRA"].max() if opsim_data else 360.0
    dec_min = opsim_data["fieldDec"].min() if opsim_data else -90.0
    dec_max = opsim_data["fieldDec"].max() if opsim_data else 33.5
    t_min = opsim_data["observationStartMJD"].min() if opsim_data else 60796.0
    t_max = opsim_data["observationStartMJD"].max() if opsim_data else 64448.0

    # Create a host galaxy anywhere on the sky.
    host = SNIaHost(
        ra=NumpyRandomFunc("uniform", low=ra_min, high=ra_max),
        dec=NumpyRandomFunc("uniform", low=dec_min, high=dec_max),
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

    sncosmo_modelname = "salt2-h17"

    source = SncosmoWrapperModel(
        sncosmo_modelname,
        t0=NumpyRandomFunc("uniform", low=t_min, high=t_max),
        x0=x0_func,
        x1=x1_func,
        c=c_func,
        ra=NumpyRandomFunc("normal", loc=host.ra, scale=0.01),
        dec=NumpyRandomFunc("normal", loc=host.dec, scale=0.01),
        redshift=host.redshift,
    )

    passbands = PassbandGroup(
        passband_parameters=[
            {
                "survey": "LSST",
                "filter_name": "u",
                "table_path": f"{passbands_dir}/LSST/u.dat",
                "units": "nm",
            },
            {
                "survey": "LSST",
                "filter_name": "r",
                "table_path": f"{passbands_dir}/LSST/r.dat",
                "units": "nm",
            },
        ],
        delta_wave=1,
    )

    res_list = []

    if phase_rest is None:
        phase_rest = np.array([-5.0, 0.0, 10.0])
    if wavelengths_rest is None:
        wavelengths_rest = np.linspace(3000, 8000, 200)

    for _n in range(0, nsample):
        res = draw_single_random_sn(
            source,
            wavelengths_rest=wavelengths_rest,
            phase_rest=phase_rest,
            passbands=passbands,
            opsim=opsim,
            opsim_data=opsim_data,
        )

        state = res["state"]
        p = {}
        for parname in ["t0", "x0", "x1", "c", "redshift", "ra", "dec"]:
            p[parname] = source.get_param(state, parname)
        for parname in ["hostmass"]:
            p[parname] = host.get_param(state, parname)
        for parname in ["distmod"]:
            p[parname] = x0_func.get_param(state, parname)
        res["parameter_values"] = p

        saltpars = {"x0": p["x0"], "x1": p["x1"], "c": p["c"], "z": p["redshift"], "t0": p["t0"]}
        model = sncosmo.Model(sncosmo_modelname)
        model.update(saltpars)
        z = p["redshift"]
        wave = passbands.waves
        if opsim is None:
            time = phase_rest * (1 + z) + p["t0"]
            assert np.allclose(res["times"], time)
        else:
            time = res["times"]

        flux_sncosmo = model.flux(time, wave)
        assert np.allclose(res["flux_flam"] * 1e10, flux_sncosmo * 1e10)

        for f in passbands.passbands:
            bandflux_sncosmo = model.bandflux(f.replace("_", ""), time, zpsys="ab", zp=8.9 + 2.5 * 9)
            assert np.allclose(res["bandfluxes"][f], bandflux_sncosmo, rtol=0.1)

        res_list.append(res)

    if return_result:
        return res_list
