import numpy as np
import sncosmo
from astropy import units as u
from scipy.interpolate import interp1d
from tdastro.astro_utils.noise_model import apply_noise
from tdastro.astro_utils.passbands import PassbandGroup
from tdastro.astro_utils.snia_utils import (
    DistModFromRedshift,
    HostmassX1Func,
    X0FromDistMod,
    num_snia_per_redshift_bin,
)
from tdastro.astro_utils.unit_utils import flam_to_fnu, fnu_to_flam
from tdastro.math_nodes.np_random import NumpyRandomFunc
from tdastro.math_nodes.scipy_random import SamplePDF
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

    ra = source.get_param(state, "ra")
    dec = source.get_param(state, "dec")
    obs_index = np.array(opsim.range_search(ra, dec, radius=1.75))

    # Update obs_index to only include observations within SN lifespan
    phase_obs = opsim["time"][obs_index] - t0
    obs_index = obs_index[(phase_obs > -20 * (1.0 + z)) & (phase_obs < 50 * (1.0 + z))]

    times = opsim["time"][obs_index].to_numpy()
    filters = opsim["filter"][obs_index].to_numpy(str)
    # Change to match band names in passbands object
    filters = np.char.add("LSST_", filters)

    if len(times) == 0:
        print(f"No overlap time in opsim for (ra,dec)=({ra:.2f},{dec:.2f})")
        return

    res["times"] = times
    res["filters"] = filters

    flux_nJy = source.evaluate(times, wave_obs, graph_state=state)

    res["flux_nJy"] = flux_nJy
    res["flux_flam"] = fnu_to_flam(
        flux_nJy,
        wave_obs,
        wave_unit=u.AA,
        flam_unit=u.erg / u.second / u.cm**2 / u.AA,
        fnu_unit=u.nJy,
    )

    res["flux_fnu"] = flux_nJy

    bandfluxes_perfect = source.get_band_fluxes(passbands, times, filters, state)
    res["bandfluxes_perfect"] = bandfluxes_perfect

    bandfluxes_error = opsim.bandflux_error_point_source(bandfluxes_perfect, obs_index)
    res["bandfluxes_error"] = bandfluxes_error
    res["bandfluxes"] = apply_noise(bandfluxes_perfect, bandfluxes_error, rng=None)

    res["state"] = state

    return res


def run_snia_end2end(oversampled_observations, passbands_dir, solid_angle=0.0001, nsample=1):
    """Test that we can sample and create SN Ia simulation using the salt3 model.

    Parameters
    ----------
    oversampled_observations : OpSim
        The opsim data to use.
    passbands_dir : str
        The name of the directory holding the passband information.
    solid_angle : float
        Solid angle for calculating number of SN.
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

    zmin = 0.1
    zmax = 0.4

    H0 = 70.0
    Omega_m = 0.3

    nsn, z = num_snia_per_redshift_bin(zmin, zmax, 100, H0=H0, Omega_m=Omega_m)
    zpdf = interp1d(z, nsn, bounds_error=False, fill_value=0)

    # Create a host galaxy.
    host = SNIaHost(
        ra=NumpyRandomFunc("uniform", low=-0.5, high=0.5),  # all pointings RA = 0.0
        dec=NumpyRandomFunc("uniform", low=-0.5, high=0.5),  # all pointings Dec = 0.0
        hostmass=NumpyRandomFunc("uniform", low=7, high=12),
        redshift=SamplePDF(zpdf),
        node_label="host",
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
        node_label="x0_func",
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
        node_label="source",
    )

    passbands = PassbandGroup(
        passband_parameters=[
            {
                "filter_name": "g",
                "table_path": f"{passbands_dir}/LSST/g.dat",
            },
            {
                "filter_name": "r",
                "table_path": f"{passbands_dir}/LSST/r.dat",
            },
        ],
        survey="LSST",
        units="nm",
        trim_quantile=0.001,
        delta_wave=1,
    )

    # Register sncosmo bandpasses
    for f, passband in passbands.passbands.items():
        sncosmo_bandpass = sncosmo.Bandpass(*passband.processed_transmission_table.T, name=f"tdastro_{f}")
        sncosmo.register(sncosmo_bandpass, force=True)

    res_list = []
    any_valid_results = False

    # Calculate nsample using SN Ia rate model
    if nsample is None and solid_angle is not None:
        nsn, _ = num_snia_per_redshift_bin(
            zmin, zmax, znbins=1, solid_angle=solid_angle, H0=H0, Omega_m=Omega_m
        )
        nsample = int(nsn)
        print(f"Drawing {nsample} samples from redshift {zmin} to {zmax}.")

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
        filters = res["filters"]

        flux_sncosmo = model.flux(time, wave)
        fnu_sncosmo = flam_to_fnu(
            flux_sncosmo,
            wave,
            wave_unit=u.AA,
            flam_unit=u.erg / u.second / u.cm**2 / u.AA,
            fnu_unit=u.nJy,
        )
        np.testing.assert_allclose(res["flux_nJy"], fnu_sncosmo, atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(res["flux_flam"], flux_sncosmo, atol=1e-30, rtol=1e-5)

        # Skip test for negative fluxes
        if np.all(flux_sncosmo > 0):
            sncosmo_band_names = np.char.add("tdastro_", filters)
            bandflux_sncosmo = model.bandflux(sncosmo_band_names, time, zpsys="ab", zp=8.9 + 2.5 * 9)
            np.testing.assert_allclose(res["bandfluxes_perfect"], bandflux_sncosmo, rtol=1e-1)

        res_list.append(res)

    assert any_valid_results, f"No valid results found over all {nsample} samples."

    return res_list, passbands
