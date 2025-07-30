import numpy as np
from astropy.cosmology import FlatLambdaCDM
from tdastro.sources.bayeSN_model import bayeSN_Model


def test_bayeSN_matchness(test_data_dir):
    """Test loading a bayeSN_Model object from a file and test we get the same
    results as the bayesian model
    """

    bayesian_flux = np.array(
        [
            [
                3.28274773e07,
                2.74288959e08,
                2.70277520e08,
                4.07166239e08,
                2.86161691e08,
                2.43041990e08,
                2.41895044e08,
                1.63319541e08,
                1.56384906e08,
                1.46499586e08,
                1.45467824e08,
                1.13228160e08,
                9.41477182e07,
                7.14146377e07,
                9.50177663e07,
                9.53728123e07,
            ],
            [
                4.24470160e06,
                1.04358573e08,
                1.53981951e08,
                2.60752039e08,
                1.62446512e08,
                1.50724949e08,
                1.76743433e08,
                1.01958387e08,
                5.71958998e07,
                3.92839096e07,
                3.79761607e07,
                3.22384974e07,
                3.60499142e07,
                6.20347574e07,
                7.25112817e07,
                7.51344153e07,
            ],
        ]
    )
    ts = np.array([0, 10])
    l = np.array(
        [
            3003.0,
            4004.0,
            5005.0,
            6006.0,
            7007.0,
            8008.0,
            9009.0,
            10010.0,
            11011.0,
            12012.0,
            13013.0,
            14014.0,
            15015.0,
            16016.0,
            17017.0,
            18018.0,
        ]
    )

    redshift = 0.001
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    distmod = cosmo.distmod(redshift).value
    m_abs = -19.5
    model = bayeSN_Model(
        theta=1,
        Av=1,
        Rv=3,
        t0=0,
        ra=0,
        dec=0,
        redshift=redshift,
        node_label="source",
        Amplitude=np.power(10.0, -0.4 * (distmod + m_abs)),
    )
    flux_density = model.evaluate(ts, l)
    assert np.allclose(bayesian_flux, flux_density, rtol=0.1)
