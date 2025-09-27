"""Wrappers for the models defined in redback.

https://github.com/nikhil-sarin/redback
https://redback.readthedocs.io/en/latest/
"""

import math

import astropy.units as uu
import numpy as np
from citation_compass import CiteClass, cite_inline

from lightcurvelynx.astro_utils.unit_utils import flam_to_fnu
from lightcurvelynx.math_nodes.bilby_priors import BilbyPriorNode
from lightcurvelynx.models.physical_model import SEDModel
from lightcurvelynx.utils.post_process_results import augment_single_lightcurve


class RedbackWrapperModel(SEDModel, CiteClass):
    """A wrapper for redback models.

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel]
    Additional parameterized values are used for specific redback models.

    References
    ----------
    * redback - https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.1203S/abstract
    * Individual models might require citation. See references in the redback documentation.

    Attributes
    ----------
    source : function
        The underlying source function that maps time + wavelength to flux.
    source_name : str
        The name used to set the source.
    source_param_names : list
        A list of the source model's parameters that we need to set.

    Parameters
    ----------
    source : str or function
        The name of the redback model function used to generate the SEDs or
        the actual function itself.
    priors : dict, bilby.prior.PriorDict, or BilbyPriorNode, optional
        The redback model's Bilby priors.
    parameters : dict, optional
        A dictionary of parameter setters to pass to the source function.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    # A class variable for the units so we are not computing them each time.
    _FLAM_UNIT = uu.erg / uu.second / uu.cm**2 / uu.AA

    def __init__(
        self,
        source,
        *,
        priors=None,
        parameters=None,
        **kwargs,
    ):
        # Check that the parameters passed in the dictionary and keyword arguments
        # do not overlap, so we only have one source of truth. This is needed for
        # parameters like `redshift` that overlap core parameters.
        if parameters is None:
            parameters = {}
        for key in parameters:
            if key in kwargs:
                raise ValueError(
                    f"Parameter '{key}' specified in both the parameters dictionary "
                    "and as a parameter itself. Please include it only in the dictionary."
                )

        super().__init__(**kwargs)

        # Add all of the items from the bilby prior node as settable parameters.
        parameters = parameters.copy()
        if priors is not None:
            if not isinstance(priors, BilbyPriorNode):
                priors = BilbyPriorNode(prior=priors)
            for param_name in priors.outputs:
                if param_name not in parameters:
                    parameters[param_name] = getattr(priors, param_name)

        # Use the parameter dictionary to create settable parameters for the model.
        # Some of these might have already been added by the superclass's constructor,
        # so we just change it.
        self.source_param_names = []
        for key, value in parameters.items():
            if key in self.setters:
                self.set_parameter(key, value)
            else:
                self.add_parameter(key, value)
            self.source_param_names.append(key)

        # Create the source itself.
        if isinstance(source, str):
            try:
                import redback
            except ImportError as err:
                raise ImportError(
                    "redback package is not installed be default. To use the RedbackWrapperModel, "
                    "please install redback. For example, you can install it with "
                    "`pip install redback`."
                ) from err

            self.source_name = source
            self.source = redback.model_library.all_models_dict[source]
        else:
            self.source_name = source.__name__
            self.source = source

        # Check if the model has a citation parameter we should include.
        if hasattr(self.source, "citation"):
            cite_inline("redback model", self.source.citation)

        # Redback models already handle redshift, so we do not want to double apply it.
        self.apply_redshift = False

    @property
    def param_names(self):
        """Return a list of the model's parameter names."""
        return self.source_param_names

    def minwave(self, graph_state=None):
        """Get the minimum wavelength of the model.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. Not used
            for this model.

        Returns
        -------
        minwave : float or None
            The minimum wavelength of the model (in angstroms) or None
            if the model does not have a defined minimum wavelength.
        """
        return 0.0

    def maxwave(self, graph_state=None):
        """Get the maximum wavelength of the model.

        Parameters
        ----------
        graph_state : GraphState, optional
            An object mapping graph parameters to their values. Not used
            for this model.

        Returns
        -------
        maxwave : float or None
            The maximum wavelength of the model (in angstroms) or None
            if the model does not have a defined maximum wavelength.
        """
        return math.inf

    def compute_sed(self, times, wavelengths, graph_state=None, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps (MJD).
        wavelengths : numpy.ndarray, optional
            A length N array of wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
           Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of SED values (in nJy).
        """
        params = self.get_local_params(graph_state)

        # Build the function arguments from the parameter values.
        fn_args = {}
        for name in self.source_param_names:
            fn_args[name] = params[name]

        # Compute the shifted times.
        t0 = params.get("t0", 0.0)
        if t0 is None:
            t0 = 0.0
        shifted_times = times - t0

        # Call the source function to get the RedbackTimeSeriesSource object.
        # We create this object with each call, because it depends on the parameters (fn_args).
        rb_result = self.source(
            shifted_times,
            output_format="sncosmo_source",
            **fn_args,
        )

        # The sncosmo-type source gives an error if the wavelengths are out of bounds, so we
        # need to use extrapolation if the wavelengths are out of bounds. The function
        # compute_sed_with_extrapolation() will trim the input wavelengths to be within the model's bounds
        # and call this function again. So the second call passes this bounds check and does not recurse.
        if np.any(wavelengths < rb_result.minwave()) or np.any(wavelengths > rb_result.maxwave()):
            return self.compute_sed_with_extrapolation(times, wavelengths, graph_state, **kwargs)

        # Query the model and convert the output to nJy.
        model_flam = rb_result.get_flux_density(shifted_times, wavelengths)
        model_fnu = flam_to_fnu(
            model_flam,
            wavelengths,
            wave_unit=uu.AA,
            flam_unit=self._FLAM_UNIT,
            fnu_unit=uu.nJy,
        )
        return model_fnu


def redback_transient_from_lightcurve(
    lightcurve,
    *,
    name="light_curve_lynx_object",
    survey_name="",
    t0=None,
    filter_invalids=True,
):
    """Create a redback Transient model from the light curve simulated by LightCurveLynx.

    Parameters
    ----------
    lightcurve : pandas.DataFrame
        The table of light curve data, such as time, filter, flux, flux_err, etc.
    name : str or function
        The name of the transient.
    survey_name : str, optional
        The name of the survey that generated the lightcurve. This is appended to the
        filter names.
    t0 : float, optional
        The t0 of the zero phase, date. This must be provided if the lightcurve
        does not already have the relative time information.
    filter_invalids : bool, optional
        If True, filter out any rows with invalid data (NaNs or Infs).

    Returns
    -------
    model : redback.transient.transient.Transient
        A Transient model representing the light curve data.
    """
    try:
        from redback.transient.transient import Transient
    except ImportError as err:
        raise ImportError(
            "redback package is not installed be default. To create redback models, "
            "please install redback. For example, you can install it with "
            "`pip install redback`."
        ) from err

    # If we don't have the augmented columns (like magg, magerr, and time_rel), add them.
    if "mag" not in lightcurve.columns:
        if t0 is None:
            raise ValueError(
                "If the lightcurve does not already have the relative time column, "
                "you must provide t0 to compute it."
            )
        lightcurve = lightcurve.copy()
        augment_single_lightcurve(lightcurve, t0=t0)

    # Filter out any invalid data points, such as NaNs. These can arise in the magnitude
    # and magnitude error columns if the flux is negative or zero.
    if filter_invalids:
        valid_mask = np.isfinite(lightcurve["mag"]) & np.isfinite(lightcurve["magerr"])
        lightcurve = lightcurve[valid_mask]

    model = Transient(
        name=name,
        data_mode="magnitude",
        time=lightcurve["time_rel"].values,
        time_err=None,
        time_mjd=lightcurve["mjd"].values,
        magnitude=lightcurve["mag"].values,
        magnitude_err=lightcurve["magerr"].values,
        bands=[f"{survey_name}{b}" for b in lightcurve["filter"]],
        active_bands="all",
        optical_data=True,
    )
    return model
