"""Models that generate the a constant SED or at all times."""

import numpy as np

from tdastro.math_nodes.given_sampler import GivenValueSampler
from tdastro.sources.physical_model import SEDModel


class StaticSEDSource(SEDModel):
    """A StaticSEDSource randomly selects an SED at each evaluation and computes
    the flux from that SED at all time steps.

    Parameterized values include:
      * dec - The object's declination in degrees. [from BasePhysicalModel]
      * distance - The object's luminosity distance in pc. [from BasePhysicalModel]
      * ra - The object's right ascension in degrees. [from BasePhysicalModel]
      * redshift - The object's redshift. [from BasePhysicalModel]
      * t0 - The t0 of the zero phase, date. [from BasePhysicalModel. Not used.]

    Attributes
    ----------
    sed_values : list
       A list of SEDs from which to sample. Each SED is represented as a
       two row numpy-array where the first row is wavelength and the
       second is flux value.

    Parameters
    ----------
    sed_values : list or numpy array.
       A single SED or a list of SEDs from which to sample. Each SED is
       represented as a two row numpy-array where the first row is wavelength
       and the second is flux value.
    weights : numpy.ndarray, optional
        A length N array indicating the relative weight from which to select
        a source at random. If None, all sources will be weighted equally.
    """

    def __init__(
        self,
        sed_values,
        weights=None,
        **kwargs,
    ):
        # If only a single SED was passed, then put it in a list by itself.
        if isinstance(sed_values, np.ndarray) and len(sed_values) == 2:
            self.sed_values = [sed_values]
        else:
            self.sed_values = sed_values

        # Validate the SED input data.
        for idx, sed in enumerate(self.sed_values):
            if not isinstance(sed, np.ndarray) or len(sed.shape) != 2 or sed.shape[0] != 2:
                raise ValueError(f"SED {idx} must be a two row numpy array of wavelength and flux.")
            if sed.shape[1] < 2:
                raise ValueError(f"SED {idx} must have at least 2 entries.")
            if not np.all(np.diff(sed[0, :]) > 0):
                raise ValueError(f"SED {idx} wavelenths are not in sorted order.")

        super().__init__(**kwargs)

        # Create a parameter that indicates which SED was sampled in each simulation.
        source_inds = [i for i in range(len(self.sed_values))]
        self._sampler_node = GivenValueSampler(source_inds, weights=weights)
        self.add_parameter("selected_idx", value=self._sampler_node, allow_gradient=False)

    def __len__(self):
        """Get the number of lightcurves."""
        return len(self.sed_values)

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
        idx = self.get_param(graph_state, "selected_idx")
        return self.sed_values[idx][0, 0]

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
        idx = self.get_param(graph_state, "selected_idx")
        return self.sed_values[idx][0, -1]

    def compute_flux(self, times, wavelengths, graph_state):
        """Draw effect-free observer frame flux densities.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame timestamps in MJD.
        wavelengths : numpy.ndarray, optional
            A length N array of observer frame wavelengths (in angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of observer frame SED values (in nJy).
        """
        # Use the SED selected by the sampler node to compute the flux density.
        model_ind = self.get_param(graph_state, "selected_idx")
        num_times = len(times)
        num_waves = len(wavelengths)

        # At each time step we interpolate SED at the query wavelengths.
        sed_fluxes = np.interp(
            wavelengths,
            self.sed_values[model_ind][0, :],
            self.sed_values[model_ind][1, :],
            left=0.0,
            right=0.0,
        )

        # We repeat the interpolated SED values at each time.
        flux_density = np.tile(sed_fluxes, num_times).reshape(num_times, num_waves)
        return flux_density
