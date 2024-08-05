from astropy.cosmology import FlatLambdaCDM
from tdastro.base_models import FunctionNode
import numpy as np
from scipy.stats.sampling import NumericalInversePolynomial
from scipy.stats import norm

class HostmassX1Distr:
    def __init__(self, hostmass):
        self.hostmass = hostmass

    def _p(self,x1,hostmass = 9.):
        if x1 < -5 or x1 > 5:
            p = 0.
        else:
            if hostmass < 10.:
                p = np.exp(-x1**2) if x1 < 0. else 1.
            else:
                p = 1.
        return p

    def pdf(self, x1):
        return self._p(x1,hostmass=self.hostmass)*norm.pdf(x1,loc=0,scale=1)


def _hostmass_x1func(hostmass):
    """Sample x1 as a function of hostmass.

    Parameters
    ----------
    hostmass : `float`
        The hostmass value.

    Returns
    -------
    x1 : `float`
        The x1 parameter in the SALT3 model
    """

    dist = HostmassX1Distr(hostmass)
    x1 = NumericalInversePolynomial(dist).rvs(1)[0]

    return x1   


def _x0_from_distmod(distmod, x1, c, alpha, beta, m_abs):
    """Calculate the SALT3 x0 parameter given distance modulus based on Tripp relation.
    distmod = -2.5*log10(x0) + alpha * x1 - beta * c - m_abs
    x0 = 10 ^ (-0.4* (distmod - alpha * x1 + beta * c + m_abs))

    Parameters
    ----------
    distmod : `float`
        The distance modulus value (in mag).
    x1 : `float`
        The SALT3 x1 parameter.
    c : `float`
        The SALT3 c parameter.
    alpha : `float`
        The alpha parameter in the Tripp relation.
    beta : `float`
        The beta parameter in the Tripp relation.
    m_abs : `float`
        The absolute magnitude of SN Ia.

    Returns
    -------
    x0 : `float`
        The x0 parameter
    """
    x0 = np.power(10., -0.4 * (distmod - alpha * x1 + beta * c + m_abs))


    return x0


def _distmod_from_redshift(redshift,H0=73.,Omega_m=0.3):
    """Compute distance modulus given redshift and cosmology.

    Parameters
    ----------
    redshift : `float`
        The redshift value.
    cosmology : `astropy.cosmology`
        The cosmology specification.

    Returns
    -------
    distmod : `float`
        The distance modulus (in mag)
    """

    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)
    distmod = cosmo.distmod(redshift).value

    return distmod


class HostmassX1Func(FunctionNode):
    """A wrapper class for the _hostmass_x1func() function.

    Attributes
    ----------
    kind : `str`
        The distance type for the Equivalency as defined by
        astropy.cosmology.units.redshift_distance.

    Parameters
    ----------
    hostmass : function or constant
        The function or constant providing the hostmass value.
    skewness : constant
        Skewness parameter that defines the skewed normal distribution.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, hostmass, **kwargs):
        # Call the super class's constructor with the needed information.
        super().__init__(
            func=_hostmass_x1func,
            hostmass=hostmass,
            **kwargs,
        )

class X0FromDistMod(FunctionNode):
    """A wrapper class for the _x0_from_distmod() function.

    Attributes
    ----------
    kind : `str`
        The distance type for the Equivalency as defined by
        astropy.cosmology.units.redshift_distance.

    Parameters
    ----------
    distmod : function or constant
        The function or constant providing the distance modulus value.
    x1 : function or constant
        The function or constant providing the x1 value.
    c : function or constant
        The function or constant providing the c value.
    alpha : function or constant
        The function or constant providing the alpha value.
    beta : function or constant
        The function or constant providing the beta value.
    m_abs : function or constant
        The function or constant providing the m_abs value.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, distmod, x1, c, alpha, beta, m_abs, **kwargs):
        # Call the super class's constructor with the needed information.
        super().__init__(
            func = _x0_from_distmod,
            distmod = distmod,
            x1 = x1,
            c = c,
            alpha = alpha,
            beta = beta,
            m_abs = m_abs,
            **kwargs,
        )

class DistModFromRedshift(FunctionNode):
    """A wrapper class for the _distmod_from_redshift() function.

    Attributes
    ----------
    kind : `str`
        The distance type for the Equivalency as defined by
        astropy.cosmology.units.redshift_distance.

    Parameters
    ----------
    redshift : function or constant
        The function or constant providing the redshift value.
    H0 : constant
        The Hubble constant.
    Omega_m : constant
        The matter density Omega_m.
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(self, redshift,H0=73.,Omega_m=0.3, **kwargs):
        # Call the super class's constructor with the needed information.
        super().__init__(
            func = _distmod_from_redshift,
            redshift = redshift,
            H0 = H0,
            Omega_m = Omega_m,
            **kwargs,
        )

