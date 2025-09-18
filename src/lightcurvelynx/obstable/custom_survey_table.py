import numpy as np

from lightcurvelynx.obstable.obs_table import ObsTable


class CustomSurveyTable(ObsTable):
    """A subclass for a completely custom survey. This can be fake data,
    ToS observations, etc.

    Parameters
    ----------
    table : dict or pandas.core.frame.DataFrame
        The table with all the ObsTable information.  Must have columns
        "time", "ra", "dec", and "filter".
    fluxerr: callable, np.array, float, dict, or str, optional
        The information needed to compute the flux error. This can be:
        - A function that takes in the bandflux (and the pandas table) and returns the flux error.
        - An array or list of flux errors that is the same length as the number of observations.
        - A constant flux error (float).
        - The column name (str) in the table that contains the flux errors.
        - A dictionary mapping filter names to a constant flux error (float).
        - None, in which case the flux error will be zero.
    colmap : dict, optional
        A mapping of standard column names to their names in the input table.
        For example, in Rubin's OpSim we might have the column "observationStartMJD"
        which maps to "time". In that case we would have an entry with key="time"
        and value="observationStartMJD".
    **kwargs : dict
        Additional keyword arguments to pass to the ObsTable constructor.  This includes overrides
        for survey parameters such as:
        - pixel_scale: The pixel scale for the camera in arcseconds per pixel.
        - radius: The angular radius of the observations (in degrees).
        - read_noise: The readout noise for the camera in electrons per pixel.
    """

    def __init__(self, table, fluxerr=None, colmap=None, **kwargs):
        super().__init__(table, colmap=colmap, **kwargs)

        # Do some basic checks on how the error is specified and extract it into a numpy array.
        if not callable(fluxerr):
            # Handle table column name.
            if isinstance(fluxerr, str):
                if fluxerr not in self._table.columns:
                    raise ValueError(f"Column {fluxerr} not found in observations table.")
                else:
                    fluxerr = self._table[fluxerr].values

            # Handle lists, arrays, constants (including None), and dictionaries
            # (with constant values per filter).
            if isinstance(fluxerr, list):
                fluxerr = np.array(fluxerr)
            elif isinstance(fluxerr, float) or isinstance(fluxerr, int):
                fluxerr = np.full(len(self._table), fluxerr)
            elif fluxerr is None:
                fluxerr = np.zeros(len(self._table))
            elif isinstance(fluxerr, dict):
                # Create an array of flux errors based on the filter column.
                all_filters = np.unique(self._table["filter"])
                for filt in all_filters:
                    if filt not in fluxerr:
                        raise ValueError(f"Flux error dictionary is missing an entry for filter {filt}.")

                fluxerr_array = np.zeros(len(self._table))
                for filt, err in fluxerr.items():
                    if not isinstance(err, (float, int)):
                        raise TypeError(
                            "When providing a dictionary of flux errors, the values must be "
                            f"constants (float or int). Found {type(err)} for filter {filt}."
                        )
                    fluxerr_array[self._table["filter"] == filt] = err
                fluxerr = fluxerr_array

            # At this point the data should be in a numpy array. Check that it is the
            # correct length and has valid values.
            if isinstance(fluxerr, np.ndarray):
                if len(fluxerr) != len(self._table):
                    raise ValueError("Flux error array must be the same length as the observations table.")
                if np.any(fluxerr < 0.0):
                    raise ValueError("Flux error must be non-negative.")
            else:
                raise TypeError(
                    "fluxerr must be a callable, np.array, float, str, list, or None. "
                    f"Found type = {type(fluxerr)}."
                )
        self.fluxerr = fluxerr

    def bandflux_error_point_source(self, bandflux, index):
        """Compute observational bandflux error for a point source

        Parameters
        ----------
        bandflux : array_like of float
            Band bandflux of the point source in nJy.
        index : array_like of int
            The index of the observation in the table.

        Returns
        -------
        flux_err : array_like of float
            Simulated bandflux noise in nJy.
        """
        # If we have precompute the flux errors into an array, use those.
        if isinstance(self.fluxerr, np.ndarray):
            return self.fluxerr[index]

        # Treat the flux error as a callable function.
        observations = self._table.iloc[index]
        return self.fluxerr(bandflux, observations)
