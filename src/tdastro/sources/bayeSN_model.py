import numpy as np
from pathlib import Path

from citation_compass import CiteClass

from astropy import units as u
import h5py
from tdastro.astro_utils.unit_utils import flam_to_fnu
from tdastro.sources.physical_model import PhysicalModel
from tdastro.effects.extinction import ExtinctionEffect
from tdastro.base_models import FunctionNode
from tdastro import _TDASTRO_BASE_DATA_DIR



class bayeSN_Model(PhysicalModel):
    """A bayeSN model

    The model is defined in (Mandel et al 2022) as:
    
    flux(time, wave) = H_grid * 10 ** (-0.4 * W_grid) * 10 ** (-0.4 * (distmod _ m_abs))

    This class is based on the bayesian implementation at:
    https://github.com/bayesn/bayesn/blob/main/bayesn/bayesn_model.py

    Parameterized values include:
        * ra - The object's right ascension in degrees. [from PhysicalModel]
        * dec - The object's declination in degrees. [from PhysicalModel]
        * redshift - The object's redshift. [from PhysicalModel]
        * t0 - The t0 of the zero phase, date. [from Physical Model]
        * Amplitude - The fixed normalisation factor. [from Amplitude class] 
        * theta - The bayeSN theta parameter.
        * Av - The bayeSN Av parameter.
        * Rv - The bayeSN Rv parameter.

    References
    ----------
    * BayeSN: Mandel S., 2020 - https://arxiv.org/pdf/2008.07538
    * M20 model: Mandel et al. 2022 (MNRAS 510, 3, 3939-3966)
    * T21 model: Thorp et al. 2021 (MNRAS 508, 3, 4310-4331) - currently unused
    * W22 model: Ward et al. 2023 (ApJ 956, 2, 111) - currently unused
    * Hsiao spectral template: Hsiao E. Y., 2009 - https://arxiv.org/abs/1503.02293

    Attributes
    ----------
    _hsiao_phase: 
        The phase for hsiao template.
    _hsiao_wave:
        The wavelengths for hsiao template.
    _hsiao_flux:
        The baseline mean intrinsic SED provided by hsiao template.
    _W0_: 
        The global W0 matrix.
    _W1_:
        The global W1 matrix.
    _l_knots_:
        The interpolation knots for wavelengths.
    _tau_knots_:
        The interpolation knots for phase.
    eps: array-like
        Set of epsilon values for each SN, describing residual colour variation.
        Currently unused.

    Parameters
    ----------
    theta: parameter
        The bayeSN theta parameter.
    Av: parameter
        The bayeSN Av parameter.
        Set of host extinction values for each SN.
    Rv: parameter
        The bayeSN Rv parameter.
        Rv valuee for host extinction.
    J_t: array-like
        Matrix for cubic spline interpolation in time axis for each SN.
    J_l: array-like
        Matrix for cubic spling interpolation in wavelength axis for each SN.
    """

    # A class variable for the units so we are not computing them each time.
    _FLAM_UNIT = u.erg / u.second / u.cm**2 / u.AA
    
    def __init__(
        self, 
        theta = None,
        Av = None,
        Rv = None,
        t0 = 0.0,
        Amplitude = 1.,
        _M20_model_path = "bayesn-model-files/BAYESN.M20",
        W0_filename = "W0.txt",
        W1_filename = "W1.txt",
        l_knots_filename = "l_knots.txt",
        tau_knots_filename = "tau_knots.txt",
        hsiao_model_path = "bayesn-model-files/hsiao.h5",
        **kwargs
    ):

        super().__init__(t0=t0, **kwargs)

        # define model specific parameters.
        self.add_parameter("theta", theta, **kwargs)
        self.add_parameter("Av", Av, **kwargs)
        self.add_parameter("Rv", Rv, **kwargs)
        self.add_parameter("Amplitude", Amplitude, **kwargs)

        # load the data files.
        self._W0_ = np.loadtxt(_TDASTRO_BASE_DATA_DIR / _M20_model_path / W0_filename)
        self._W1_ = np.loadtxt(_TDASTRO_BASE_DATA_DIR / _M20_model_path / W1_filename)
        self._l_knots_ = np.loadtxt(_TDASTRO_BASE_DATA_DIR / _M20_model_path / l_knots_filename)
        self._tau_knots_ = np.loadtxt(_TDASTRO_BASE_DATA_DIR / _M20_model_path / tau_knots_filename)
        with h5py.File(_TDASTRO_BASE_DATA_DIR / hsiao_model_path, 'r') as file:
            data = file['default']
            self._hsiao_phase = data['phase'][()].astype('float64')
            self._hsiao_wave = data['wave'][()].astype('float64')
            self._hsiao_flux = data['flux'][()].astype('float64')


    # HELPER FUNCTIONs:
    def compute_invKD(self, x):
        """
        Compute the operator matrix K^{-1}D to get second derivatives for natural cubic spline.
    
        Parameters
        ----------
        x : (n,) array_like
            Knot positions (non-uniform, strictly increasing).
    
        Returns
        -------
        invKD : (n, n) ndarray
            Matrix such that M = invKD @ y gives second derivatives of y.
        """
        n = len(x)
        K = np.zeros((n-2, n-2))
        D = np.zeros((n-2, n))
    
        for j in range(1, n-1):
            i = j - 1
            h0 = x[j] - x[j-1]
            h1 = x[j+1] - x[j]
            if i > 0:
                K[i, i-1] = h0 / 6
            K[i, i] = (h0 + h1) / 3
            if i < n - 3:
                K[i, i+1] = h1 / 6
    
        for j in range(1, n-1):
            i = j - 1
            h0 = x[j] - x[j-1]
            h1 = x[j+1] - x[j]
            D[i, j-1] = 1 / h0
            D[i, j]   = -1 / h0 - 1 / h1
            D[i, j+1] = 1 / h1
    
        invKD = np.zeros((n, n))
        invKD[1:-1, :] = np.linalg.solve(K, D)
        return invKD
    
    def natural_cubic_spline_basis_matrix_from_invKD(self, x, xq_array, invKD):
        """
        Compute basis matrix J for multiple query points using precomputed second derivative matrix.
    
        Parameters
        ----------
        x : (n,) array_like
            Knot positions.
        xq_array : (m,) array_like
            Query points.
        invKD : (n, n) array_like
            Precomputed matrix such that second_derivatives = invKD @ y.
    
        Returns
        -------
        J : (m, n) ndarray
            Basis matrix. Each row J[i, :] is the spline basis vector for xq_array[i].
        """
        x = np.asarray(x)
        xq_array = np.asarray(xq_array)
        n = len(x)
        m = len(xq_array)
    
        J = np.zeros((m, n))
        idxs = np.searchsorted(x, xq_array) - 1
        idxs = np.clip(idxs, 0, n - 2)
    
        x0 = x[idxs]
        x1 = x[idxs + 1]
        h = x1 - x0
        a = (x1 - xq_array) / h
        b = 1 - a
        c = ((a**3 - a) * h**2) / 6
        d = ((b**3 - b) * h**2) / 6
    
        rows = np.arange(m)
        J[rows, idxs] = a
        J[rows, idxs + 1] = b
    
        J += c[:, None] * invKD[idxs] + d[:, None] * invKD[idxs + 1]
        return J

    def compute_second_derivatives_1d(self, x, y):
        """
        Compute natural cubic spline second derivatives (M) for 1D input.
    
        Parameters
        ----------
        x : (n,) array_like
            Knot positions.
        y : (n,) array_like
            Values at knots.
    
        Returns
        -------
        M : (n,) ndarray
            Second derivatives at knots.
        """
        n = len(x)
        h = np.diff(x)
        alpha = np.zeros(n)
        for i in range(1, n - 1):
            alpha[i] = (3/h[i])*(y[i+1] - y[i]) - (3/h[i-1])*(y[i] - y[i-1])
        l = np.ones(n)
        mu = np.zeros(n)
        z = np.zeros(n)
        for i in range(1, n - 1):
            l[i] = 2*(x[i+1] - x[i-1]) - h[i-1]*mu[i-1]
            mu[i] = h[i]/l[i]
            z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]
        M = np.zeros(n)
        for j in range(n - 2, 0, -1):
            M[j] = z[j] - mu[j]*M[j+1]
        return M
    
    def compute_2d_second_derivatives(self, x, y, z):
        """
        Compute second derivatives in both x and y directions for 2D natural cubic spline.
    
        Parameters
        ----------
        x : (n,) array_like
            Knot positions in x.
        y : (m,) array_like
            Knot positions in y.
        z : (n, m) array_like
            Function values on 2D grid.
    
        Returns
        -------
        Mx : (n, m) ndarray
            Second derivatives with respect to x.
        My : (n, m) ndarray
            Second derivatives with respect to y.
        """
        n, m = len(x), len(y)
        Mx = np.zeros((n, m))
        My = np.zeros((n, m))
        for j in range(m):
            Mx[:, j] = self.compute_second_derivatives_1d(x, z[:, j])
        for i in range(n):
            My[i, :] = self.compute_second_derivatives_1d(y, z[i, :])
        return Mx, My
    
    def evaluate_natural_spline_2d_vectorized(self, x, y, z, Mx, My, xq, yq):
        """
        Vectorized 2D natural cubic spline evaluation using second derivatives.
    
        Parameters
        ----------
        x : (n,) array_like
            Knot positions in x.
        y : (m,) array_like
            Knot positions in y.
        z : (n, m) array_like
            Function values on 2D grid.
        Mx : (n, m) array_like
            Second derivatives w.r.t. x.
        My : (n, m) array_like
            Second derivatives w.r.t. y.
        xq : (p,) array_like
            Query positions in x.
        yq : (q,) array_like
            Query positions in y.
    
        Returns
        -------
        Zq : (p, q) ndarray
            Interpolated 2D surface values.
        """
        xq = np.atleast_1d(xq)
        yq = np.atleast_1d(yq)
        nx, ny = len(x), len(y)
        px, py = len(xq), len(yq)
    
        xi_idx = np.clip(np.searchsorted(x, xq) - 1, 0, nx - 2)
        yj_idx = np.clip(np.searchsorted(y, yq) - 1, 0, ny - 2)
    
        Zq = np.zeros((px, py))
    
        for j in range(py):
            y0_idx = yj_idx[j]
            y1_idx = y0_idx + 1
            hy = y[y1_idx] - y[y0_idx]
            ay = (y[y1_idx] - yq[j]) / hy
            by = 1 - ay
    
            f_y0 = []
            f_y1 = []
    
            for i in range(px):
                x0_idx = xi_idx[i]
                x1_idx = x0_idx + 1
                hx = x[x1_idx] - x[x0_idx]
                ax = (x[x1_idx] - xq[i]) / hx
                bx = 1 - ax
    
                # Values at 4 corners
                z00 = z[x0_idx, y0_idx]
                z10 = z[x1_idx, y0_idx]
                z01 = z[x0_idx, y1_idx]
                z11 = z[x1_idx, y1_idx]
    
                Mx00 = Mx[x0_idx, y0_idx]
                Mx10 = Mx[x1_idx, y0_idx]
                Mx01 = Mx[x0_idx, y1_idx]
                Mx11 = Mx[x1_idx, y1_idx]
    
                # Spline in x for fixed y0 and y1
                fx0 = ((ax**3 * Mx00 + bx**3 * Mx10) * hx / 6 +
                       (ax * (z00 - Mx00 * hx**2 / 6) + bx * (z10 - Mx10 * hx**2 / 6)))
                fx1 = ((ax**3 * Mx01 + bx**3 * Mx11) * hx / 6 +
                       (ax * (z01 - Mx01 * hx**2 / 6) + bx * (z11 - Mx11 * hx**2 / 6)))
    
                f_y0.append(fx0)
                f_y1.append(fx1)
    
            f_y0 = np.array(f_y0)
            f_y1 = np.array(f_y1)
    
            # Spline in y using fx0 and fx1
            My0 = My[xi_idx, y0_idx]
            My1 = My[xi_idx, y1_idx]
    
            Zq[:, j] = ((ay**3 * My0 + by**3 * My1) * hy / 6 +
                        (ay * (f_y0 - My0 * hy**2 / 6) + by * (f_y1 - My1 * hy**2 / 6)))
    
        return Zq

    def evaluate_2d_cubic_spline(self, x, y, z, xq, yq):
        """
        Evaluate 2D natural cubic spline at given query points with auto second derivative computation.
    
        Parameters
        ----------
        x : (n,) array_like
            Knot positions in x direction.
        y : (m,) array_like
            Knot positions in y direction.
        z : (n, m) array_like
            Function values at grid points.
        xq : (p,) array_like
            Query points in x.
        yq : (q,) array_like
            Query points in y.
    
        Returns
        -------
        Zq : (p, q) ndarray
            Interpolated values at query grid.
        """
        Mx, My = self.compute_2d_second_derivatives(x, y, z)
        return self.evaluate_natural_spline_2d_vectorized(x, y, z, Mx, My, xq, yq)

    def mask_by_time(self, times, graph_state=None):
        """Compute a mask for whether a given time is of interest for a given object.
        For example, a user can use this function to generate a mask to include
        only the observations of interest for a window around the supernova.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        graph_state : GraphState, optional
            An object mapping graph parameters to their values.

        Returns
        -------
        time_mask : numpy.ndarray
            A length T array of Booleans indicating whether the time is of interest.
        """
        if graph_state is None:
            raise ValueError("graph_state needed to compute mask_by_time")

        z = self.get_param(graph_state, "redshift", 0.0)
        if z is None:
            z = 0.0

        t0 = self.get_param(graph_state, "t0", 0.0)
        if t0 is None:
            t0 = 0.0

        # Compute the mask.
        good_times = (times > t0 + -20.0 * (1.0 + z)) & (times < t0 + 50.0 * (1.0 + z))
        return good_times
    
    # MAIN FUNCTION:
    def compute_flux(self, times, wavelength, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps in MJD.
        wavelengths : numpy.ndarray, optional
            A length N array of rest frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
            
        **kwargs : dict, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        
        W_0 = self._W0_
        W_1 = self._W1_
        l_knots = self._l_knots_
        tau_knots = self._tau_knots_
        hsiao_wave = self._hsiao_wave
        hsiao_phase = self._hsiao_phase
        hsiao_flux = self._hsiao_flux
        params = self.get_local_params(graph_state)

        l = wavelength
        tau = times - params["t0"]
        #l = np.linspace(3000, 8000, 200)
        #tau = np.linspace(-20, 50, 71)
        W = W_0 + params["theta"] * W_1
        invKD_l = self.compute_invKD(l_knots)
        J_l = self.natural_cubic_spline_basis_matrix_from_invKD(l_knots, l, invKD_l)
        invKD_tau = self.compute_invKD(tau_knots)
        J_t = self.natural_cubic_spline_basis_matrix_from_invKD(tau_knots, tau, invKD_tau)
        J_t_T = np.atleast_2d(J_t).T
        WJt = np.matmul(W, J_t_T)
        W_grid = np.matmul(J_l, WJt)
        W_grid = np.atleast_2d(W_grid).T
        H_grid = self.evaluate_2d_cubic_spline(hsiao_phase, hsiao_wave, hsiao_flux, tau, l)
        flux_density = H_grid * 10 ** (-0.4 * W_grid)


        # Apply dust extinction law
        # Get ebv such that ebv = Rv/Av
        ebv = params["Av"]/params["Rv"]
        ext = ExtinctionEffect(extinction_model="F99", ebv=ebv)
        flux_density = ext.apply(flux_density, tau, l, ebv)

        # Apply the fixed normalisation factor effect
        flux_density = flux_density * params["Amplitude"]
        
        # Convert to the correct units.
        flux_density = flam_to_fnu(
            flux_density,
            wavelength,
            wave_unit=u.AA,
            flam_unit=self._FLAM_UNIT,
            fnu_unit=u.nJy,
        )

        return flux_density


class AmplitudeFromDistMod(FunctionNode):
    def __init__(self, distmod,m_abs, **kwargs):
        # Call the super class's constructor with the needed information.
        super().__init__(
            func=self._amplitude_given_distmod,
            distmod=distmod,
            m_abs=m_abs,
            **kwargs,
        )

    def _amplitude_given_distmod(self, distmod, m_abs):
        """Compute distance modulus given redshift and cosmology.

        Parameters
        ----------
        redshift : float or numpy.ndarray
            The redshift value(s).

        Returns
        -------
        distmod : float or numpy.ndarray
            The distance modulus (in mag)
        """
        return np.power(10.,-0.4*(distmod + m_abs))