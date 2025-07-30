import numpy as np
from astropy import units as u
from tdastro.astro_utils.unit_utils import fnu_to_flam
from astropy.cosmology import FlatLambdaCDM
from tdastro.math_nodes.np_random import NumpyRandomFunc
from tdastro.sources.bayeSN_model import bayeSN_Model

def test_bayeSN_matchness(test_data_dir):
    """Test loading a bayeSN_Model object from a file and test we get the same 
    results as the bayesian model
    """

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
    ts = np.array([0,10])
    l = np.array([ 3003.,  4004.,  5005.,  6006.,  7007.,  8008.,  9009., 10010.,
       11011., 12012., 13013., 14014., 15015., 16016., 17017., 18018.])
    
    redshift = 0.001
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    distmod=cosmo.distmod(redshift).value
    m_abs=-19.5
    model = bayeSN_Model(theta = 1,
                         Av = 1,
                         Rv = 3,
                         t0=0,
                         ra=0,
                         dec=0,
                         redshift=redshift,
                         node_label="source",
                         Amplitude=np.power(10.,-0.4*(distmod + m_abs)),
    )
    flux_density = model.evaluate(ts, l)
    assert np.allclose(bayesian_flux, flux_density, rtol=0.1)
    