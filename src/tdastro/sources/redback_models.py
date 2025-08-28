"""Wrappers for the models defined in redback.

https://github.com/nikhil-sarin/redback
https://redback.readthedocs.io/en/latest/
"""

import math

import numpy as np
from citation_compass import CiteClass

from tdastro.sources.physical_model import PhysicalModel


class RedbackWrapperModel(PhysicalModel, CiteClass):
    """A wrapper for redback models.

    Parameterized values include:
      * dec - The object's declination in degrees. [from PhysicalModel]
      * distance - The object's luminosity distance in pc. [from PhysicalModel]
      * ra - The object's right ascension in degrees. [from PhysicalModel]
      * redshift - The object's redshift. [from PhysicalModel]
      * t0 - The t0 of the zero phase, date. [from PhysicalModel]
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
    parameters : dict, optional
        A dictionary of parameter setters to pass to the source function.
    **kwargs : dict, optional
        Any additional keyword arguments.
    """

    def __init__(
        self,
        source,
        *,
        parameters=None,
        **kwargs,
    ):
        try:
            import redback
        except ImportError as err:
            raise ImportError(
                "redback package is not installed be default. To use the RedbackWrapperModel, "
                "please install redback. For example, you can install it with "
                "`pip install redback` or `conda install conda-forge::redback`."
            ) from err

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
            self.source_name = source
            self.source = redback.model_library.all_models_dict[source]
        else:
            self.source_name = source.__name__
            self.source = source

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

    def compute_flux(self, times, wavelengths, graph_state=None, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of rest frame timestamps.
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

        # Compute the results. Redback returns a value per wavelength, so we iterate
        # over the wavelengths.
        results = np.zeros((len(times), len(wavelengths)))
        for i, wave in enumerate(wavelengths):
            results[:, i] = self.source(
                shifted_times,
                frequency=wave,
                output_format="flux_density",
                **fn_args,
            )
        return results
