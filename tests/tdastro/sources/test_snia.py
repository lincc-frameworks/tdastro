import numpy as np
import sncosmo
from astropy import units as u
from tdastro.astro_utils.passbands import PassbandGroup
from tdastro.astro_utils.snia_utils import DistModFromRedshift, HostmassX1Func, X0FromDistMod
from tdastro.astro_utils.unit_utils import flam_to_fnu
from tdastro.rand_nodes.np_random import NumpyRandomFunc
from tdastro.sources.sncomso_models import SncosmoWrapperModel
from tdastro.sources.snia_host import SNIaHost


def draw_single_random_sn(
    source,
    opsim,
    passbands,
):
    """
    Draw a single random SN realiztion
    """

    state = source.sample_parameters()

    z = source.get_param(state, "redshift")
    wave_obs = passbands.waves
    wavelengths_rest = wave_obs / (1.0 + z)

    res = {"wavelengths_rest": wavelengths_rest}

    t0 = source.get_param(state, "t0")

    if opsim:
        ra = source.get_param(state, "ra")
        dec = source.get_param(state, "dec")
        obs = opsim.get_observations(ra, dec, radius=1.75, cols=["time", "filter"])

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


def run_snia_end2end(oversampled_observations, passbands_dir, nsample=1):
    """Test that we can sample and create SN Ia simulation using the salt3 model.

    Parameters
    ----------
    oversampled_observations : OpSim
        The opsim data to use.
    passbands_dir : str
        The name of the directory holding the passband information.
    nsample : int
        The number of samples to test.
        Default:  1

    Returns
    -------
    res_list : dict
        A dictionary of lists of sampling and result information.
    passbands : PassbandGroup
        The passbands used.
    """
    t_min = oversampled_observations["observationStartMJD"].min()
    t_max = oversampled_observations["observationStartMJD"].max()

    # Create a host galaxy.
    host = SNIaHost(
        ra=NumpyRandomFunc("uniform", low=-0.5, high=0.5),  # all pointings RA = 0.0
        dec=NumpyRandomFunc("uniform", low=-0.5, high=0.5),  # all pointings Dec = 0.0
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

    sncosmo_modelname = "salt3"

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
                "filter_name": "r",
                "table_path": f"{passbands_dir}/LSST/r.dat",
            },
            {
                "filter_name": "i",
                "table_path": f"{passbands_dir}/LSST/u.dat",
            },
        ],
        survey="LSST",
        units="nm",
        trim_quantile=0.001,
        delta_wave=1,
    )

    res_list = []
    any_valid_results = False
    for _n in range(0, nsample):
        res = draw_single_random_sn(
            source,
            opsim=oversampled_observations,
            passbands=passbands,
        )

        if res is None:
            continue
        any_valid_results = True

        state = res["state"]

        p = {}
        for parname in ["t0", "x0", "x1", "c", "redshift", "ra", "dec"]:
            p[parname] = float(source.get_param(state, parname))
        for parname in ["hostmass"]:
            p[parname] = host.get_param(state, parname)
        for parname in ["distmod"]:
            p[parname] = x0_func.get_param(state, parname)
        res["parameter_values"] = p

        saltpars = {"x0": p["x0"], "x1": p["x1"], "c": p["c"], "z": p["redshift"], "t0": p["t0"]}
        model = sncosmo.Model(sncosmo_modelname)
        model.update(saltpars)
        wave = passbands.waves
        time = res["times"]

        flux_sncosmo = model.flux(time, wave)
        np.testing.assert_allclose(res["flux_flam"], flux_sncosmo, atol=1e-30, rtol=1e-5)

        for f, passband in passbands.passbands.items():
            # Skip test for negative fluxes
            if np.any(flux_sncosmo < 0):
                continue
            sncosmo_band = sncosmo.Bandpass(*passband.processed_transmission_table.T, name=f)
            bandflux_sncosmo = model.bandflux(sncosmo_band, time, zpsys="ab", zp=8.9 + 2.5 * 9)
            np.testing.assert_allclose(res["bandfluxes"][f], bandflux_sncosmo, rtol=1e-1, err_msg=f"band {f}")

        res_list.append(res)

    assert any_valid_results, f"No valid results found over all {nsample} samples."

    return res_list, passbands


def test_snia_end2end(oversampled_observations, passbands_dir):
    """Test that the end to end run works."""
    num_samples = 1
    res_list, passbands = run_snia_end2end(oversampled_observations, passbands_dir, nsample=num_samples)
    assert len(res_list) == num_samples
    assert len(passbands) == 2
