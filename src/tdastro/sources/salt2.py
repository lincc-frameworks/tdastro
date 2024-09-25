import os.path

import numpy as np

from tdastro.sources.physical_model import PhysicalModel
from tdastro.utils.bicubic_interp import BicubicInterpolator


class SALT2JaxModel(PhysicalModel):
    """A SALT2 model implemented with JAX for it can use auto-differentiation.

    The model is defined in (Guy J., 2007) as:

    flux(time, wave) = x0 * [M0(time, wave) + x1 * M1(time, wave)] * exp(c * CL(wave))

    where x0, x1, and c are given parameters, M0 is the average spectral sequence,
    M1 is the first compoment to describe variability, and CL is the average color
    correction law.

    Based on the SNCosmo implementation at:
    https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py

    Attributes
    ----------
    _m0_model : BicubicInterpolator
        The interpolator for the m0 parameter.
    _m1_model : BicubicInterpolator
        The interpolator for the m1 parameter.

    Parameters
    ----------
    x0 : parameter
        The SALT2 x0 parameter.
    x1 : parameter
        The SALT2 x1 parameter.
    c : parameter
        The SALT2 c parameter.
    model_dir : `str`
        The path for the model file directory.
        Default: ""
    m0_filename : `str`
        The file name for the m0 model component.
        Default: "salt2_template_0.dat"
    m1_filename : `str`
        The file name for the m1 model component.
        Default: "salt2_template_1.dat"
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        x0=None,
        x1=None,
        c=None,
        model_dir="",
        m0_filename="salt2_template_0.dat",
        m1_filename="salt2_template_1.dat",
    ):
        super().__init__(**kwargs)

        # Add the model specific parameters.
        self.add_parameter("x0", x0, **kwargs)
        self.add_parameter("x1", x0, **kwargs)
        self.add_parameter("c", x0, **kwargs)

        # Load the data files.
        self._m0_model = BicubicInterpolator.from_grid_file(os.path.join(model_dir, m0_filename))
        self._m1_model = BicubicInterpolator.from_grid_file(os.path.join(model_dir, m1_filename))

    def _evaluate(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : `numpy.ndarray`
            A length T array of rest frame timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths (in angstroms).
        graph_state : `GraphState`
            An object mapping graph parameters to their values.
        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values (in nJy).
        """
        m0 = self._m0_model(phase, wave)
        m1 = self._m1_model(phase, wave)
        return (
            self._parameters[0]
            * (m0 + self._parameters[1] * m1)
            * 10.0 ** (-0.4 * self._colorlaw(wave) * self._parameters[2])
        )

        flux_density = np.zeros((len(times), len(wavelengths)))
        params = self.get_local_params(graph_state)

        time_mask = (times >= params["t0"]) & (times <= params["t1"])
        flux_density[time_mask] = params["brightness"]
        return flux_density


class SALT2Source(Source):
    """The SALT2 Type Ia supernova spectral timeseries model.

    The spectral flux density of this model is given by

    .. math::

       F(t, \\lambda) = x_0 (M_0(t, \\lambda) + x_1 M_1(t, \\lambda))
                       \\times 10^{-0.4 CL(\\lambda) c}

    where ``x0``, ``x1`` and ``c`` are the free parameters of the model,
    ``M_0``, ``M_1`` are the zeroth and first components of the model, and
    ``CL`` is the colorlaw, which gives the extinction in magnitudes for
    ``c=1``.

    Parameters
    ----------
    modeldir : str, optional
        Directory path containing model component files. Default is `None`,
        which means that no directory is prepended to filenames when
        determining their path.
    m0file, m1file, clfile : str or fileobj, optional
        Filenames of various model components. Defaults are:

        * m0file = 'salt2_template_0.dat' (2-d grid)
        * m1file = 'salt2_template_1.dat' (2-d grid)
        * clfile = 'salt2_color_correction.dat'

    errscalefile, lcrv00file, lcrv11file, lcrv01file, cdfile : str or fileobj
        (optional) Filenames of various model components for
        model covariance in synthetic photometry. See
        ``bandflux_rcov`` for details.  Defaults are:

        * errscalefile = 'salt2_lc_dispersion_scaling.dat' (2-d grid)
        * lcrv00file = 'salt2_lc_relative_variance_0.dat' (2-d grid)
        * lcrv11file = 'salt2_lc_relative_variance_1.dat' (2-d grid)
        * lcrv01file = 'salt2_lc_relative_covariance_01.dat' (2-d grid)
        * cdfile = 'salt2_color_dispersion.dat' (1-d grid)

    Notes
    -----
    The "2-d grid" files have the format ``<phase> <wavelength>
    <value>`` on each line.

    The phase and wavelength values of the various components don't
    necessarily need to match. (In the most recent salt2 model data,
    they do not all match.) The phase and wavelength values of the
    first model component (in ``m0file``) are taken as the "native"
    sampling of the model, even though these values might require
    interpolation of the other model components.

    """

    # These files are distributed with SALT2 model data but not currently
    # used:
    # v00file = 'salt2_spec_variance_0.dat'              : 2dgrid
    # v11file = 'salt2_spec_variance_1.dat'              : 2dgrid
    # v01file = 'salt2_spec_covariance_01.dat'           : 2dgrid

    _param_names = ["x0", "x1", "c"]
    param_names_latex = ["x_0", "x_1", "c"]
    _SCALE_FACTOR = 1e-12

    def __init__(
        self,
        modeldir=None,
        m0file="salt2_template_0.dat",
        m1file="salt2_template_1.dat",
        clfile="salt2_color_correction.dat",
        cdfile="salt2_color_dispersion.dat",
        errscalefile="salt2_lc_dispersion_scaling.dat",
        lcrv00file="salt2_lc_relative_variance_0.dat",
        lcrv11file="salt2_lc_relative_variance_1.dat",
        lcrv01file="salt2_lc_relative_covariance_01.dat",
        name=None,
        version=None,
    ):
        self.name = name
        self.version = version
        self._model = {}
        self._parameters = np.array([1.0, 0.0, 0.0])

        names_or_objs = {
            "M0": m0file,
            "M1": m1file,
            "LCRV00": lcrv00file,
            "LCRV11": lcrv11file,
            "LCRV01": lcrv01file,
            "errscale": errscalefile,
            "cdfile": cdfile,
            "clfile": clfile,
        }

        # Make filenames into full paths.
        if modeldir is not None:
            for k in names_or_objs:
                v = names_or_objs[k]
                if v is not None and isinstance(v, str):
                    names_or_objs[k] = os.path.join(modeldir, v)

        # model components are interpolated to 2nd order
        for key in ["M0", "M1"]:
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
            values *= self._SCALE_FACTOR
            self._model[key] = BicubicInterpolator(phase, wave, values)

            # The "native" phases and wavelengths of the model are those
            # of the first model component.
            if key == "M0":
                self._phase = phase
                self._wave = wave

        # model covariance is interpolated to 1st order
        for key in ["LCRV00", "LCRV11", "LCRV01", "errscale"]:
            phase, wave, values = read_griddata_ascii(names_or_objs[key])
            self._model[key] = BicubicInterpolator(phase, wave, values)

        # Set the colorlaw based on the "color correction" file.
        self._set_colorlaw_from_file(names_or_objs["clfile"])

        # Set the color dispersion from "color_dispersion" file
        w, val = np.loadtxt(names_or_objs["cdfile"], unpack=True)
        self._colordisp = Spline1d(w, val, k=1)  # linear interp.
