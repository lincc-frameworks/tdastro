import numpy as np
import pytest
from citation_compass import find_in_citations
from astropy import units as u
from tdastro.astro_utils.unit_utils import fnu_to_flam
from astropy.cosmology import FlatLambdaCDM
from tdastro.math_nodes.np_random import NumpyRandomFunc
from tdastro.sources.bayeSN_model import bayesnModel


def test_bayesn_matchness(test_data_dir):
    """Test loading a bayesnModel object from a file and test we get the same 
    results as the bayesian model
    """
    # We get the bayesian_flux from running the following M20 model code provided 
    # by bayesian model
        # from bayesn.bayesn_model import SEDmodel
        # model = SEDmodel(load_model='M20_model', 
        #                  filter_yaml='../data/passbands/LSST/filters.yaml')
        # ts =  np.linspace(-20, 50, 71) # Set of phases at which to generate spectra 
        # for each object
        # ts = np.array([0,10])
        # N = 1 # Number of SNe to generate spectra for. 
        # If you specify parameter values, the number of values passed
        # needs to match N e.g. if you want spectra for 10 objects, 
        # specify 10 theta values (unless you want to use the same value 
        # for all of them, in which case you can just pass one single value)
        # sim = model.simulate_spectrum(ts, N, dl=1000 ,del_M=0, z=0.001, mu=0, 
        #                               ebv_mw=0, theta=1, AV=1, RV=3, eps=0)
        # l, spec, params = sim

    bayesian_flux = np.array(
        [[3.28274773e+07, 2.74288959e+08, 2.70277520e+08, 4.07166239e+08,
        2.86161691e+08, 2.43041990e+08, 2.41895044e+08, 1.63319541e+08,
        1.56384906e+08, 1.46499586e+08, 1.45467824e+08, 1.13228160e+08,
        9.41477182e+07, 7.14146377e+07, 9.50177663e+07, 9.53728123e+07],
       [4.24470160e+06, 1.04358573e+08, 1.53981951e+08, 2.60752039e+08,
        1.62446512e+08, 1.50724949e+08, 1.76743433e+08, 1.01958387e+08,
        5.71958998e+07, 3.92839096e+07, 3.79761607e+07, 3.22384974e+07,
        3.60499142e+07, 6.20347574e+07, 7.25112817e+07, 7.51344153e+07]]
    )
    times = np.array([0,10])
    wavelengths = np.array([ 3003.,  4004.,  5005.,  6006.,  7007.,  8008.,  9009., 10010.,
       11011., 12012., 13013., 14014., 15015., 16016., 17017., 18018.])
    
    redshift = 0.001
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    distmod=cosmo.distmod(redshift).value
    m_abs=-19.5
    model = bayesnModel(theta=1,
                         Av=1,
                         Rv=3,
                         t0=0,
                         ra=0,
                         dec=0,
                         redshift=redshift,
                         node_label="source",
                         _M20_model_path = test_data_dir / "bayesn-model-files/BAYESN.M20",
                         hsiao_model_path = test_data_dir / "bayesn-model-files/hsiao.h5",
                         Amplitude=np.power(10.,-0.4*(distmod + m_abs)),
    )
    flux_density = model.evaluate(times, wavelengths)
    assert np.allclose(bayesian_flux, flux_density, rtol=0.1)

def test_bayesn_no_model(test_data_dir):
    """Test that we fail if using the wrong model directory."""
    dir_name = test_data_dir / "no_such_salt2_model_dir"
    with pytest.raises(FileNotFoundError):
        redshift = 0.001
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        distmod=cosmo.distmod(redshift).value
        m_abs=-19.5
        _ = bayesnModel(theta=1,
                         Av=1,
                         Rv=3,
                         t0=0,
                         ra=0,
                         dec=0,
                         redshift=redshift,
                         node_label="source",
                         Amplitude=np.power(10.,-0.4*(distmod + m_abs)),
                         _M20_model_path=dir_name)


def test_bayesn_citation():
    """Test the citations for the bayesian model."""
    bayesn_citations = find_in_citations("bayesnModel")
    for citation in bayesn_citations:
        assert "Mandel S., 2020" in citation
        assert "Hsiao E. Y., 2009" in citation

    