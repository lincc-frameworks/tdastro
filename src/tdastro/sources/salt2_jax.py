from pathlib import Path

from astropy import units as u

from tdastro.astro_utils.salt2_color_law import SALT2ColorLaw
from tdastro.astro_utils.unit_utils import flam_to_fnu
from tdastro.sources.physical_model import PhysicalModel
from tdastro.utils.bicubic_interp import BicubicInterpolator


class SALT2JaxModel(PhysicalModel):
    """A SALT2 model implemented with JAX for it can use auto-differentiation.

    The model is defined in (Guy J., 2007) as:

    flux(time, wave) = x0 * [M0(time, wave) + x1 * M1(time, wave)] * exp(c * CL(wave))

    where x0, x1, and c are given parameters, M0 is the average spectral sequence,
    M1 is the first compoment to describe variability, and CL is the average color
    correction law.

    We use the formulation in sncosmo where CL is defined such that:

    flux(time, wave) = x0 * [M0(time, wave) + x1 * M1(time, wave)] * 10 ** (-0.4 * c * CL(wave))

    This class is based on the sncosmo implementation at:
    https://github.com/sncosmo/sncosmo/blob/v2.10.1/sncosmo/models.py
    The wrapped sncosmo version in sncosmo_models.py is faster and should be used
    when auto-differentiation is not needed.

    Attributes
    ----------
    _m0_model : BicubicInterpolator
        The interpolator for the m0 parameter.
    _m1_model : BicubicInterpolator
        The interpolator for the m1 parameter.
    _colorlaw : SALT2ColorLaw
        The data to apply the color law.

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
    cl_filename : `str`
        The file name of the color law correction coefficients.
        Default: "salt2_color_correction.dat",
    **kwargs : `dict`, optional
        Any additional keyword arguments.
    """

    # A class variable for the units so we are not computing them each time.
    _FLAM_UNIT = u.erg / u.second / u.cm**2 / u.AA

    def __init__(
        self,
        x0=None,
        x1=None,
        c=None,
        model_dir="",
        m0_filename="salt2_template_0.dat",
        m1_filename="salt2_template_1.dat",
        cl_filename="salt2_color_correction.dat",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Add the model specific parameters.
        self.add_parameter("x0", x0, **kwargs)
        self.add_parameter("x1", x1, **kwargs)
        self.add_parameter("c", c, **kwargs)

        # Load the data files.
        model_path = Path(model_dir)
        self._m0_model = BicubicInterpolator.from_grid_file(
            model_path / m0_filename,
            scale_factor=1e-12,
        )
        self._m1_model = BicubicInterpolator.from_grid_file(
            model_path / m1_filename,
            scale_factor=1e-12,
        )

        # Use the default color correction values.
        self._colorlaw = SALT2ColorLaw.from_file(model_path / cl_filename)

    def compute_flux(self, phase, wavelengths, graph_state, use_flam=False, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        phase : `numpy.ndarray`
            A length T array of rest frame timestamps.
        wavelengths : `numpy.ndarray`, optional
            A length N array of wavelengths (in angstroms).
        graph_state : `GraphState`
            An object mapping graph parameters to their values.
        use_flam : bool
            Output the results in f_lambda units as opposed to f_nu units.
            This should only be set to True for specific testing scenarios.

        **kwargs : `dict`, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : `numpy.ndarray`
            A length T x N matrix of SED values (in nJy by default).
        """
        m0_vals = self._m0_model(phase, wavelengths)
        m1_vals = self._m1_model(phase, wavelengths)
        params = self.get_local_params(graph_state)

        flux_density = (
            params["x0"]
            * (m0_vals + params["x1"] * m1_vals)
            * 10.0 ** (-0.4 * self._colorlaw.apply(wavelengths) * params["c"])
        )

        # Convert to f_nu. Doing this conversion should be the desired behavior
        # almost all of the time.
        if not use_flam:
            flux_density = flam_to_fnu(
                flux_density,
                wavelengths,
                wave_unit=u.AA,
                flam_unit=self._FLAM_UNIT,
                fnu_unit=u.nJy,
            )
        return flux_density