import numpy as np
import pandas as pd
import pytest

from lightcurvelynx.obstable.custom_survey_table import CustomSurveyTable


def test_create_custom_survey_table_given():
    """Create a minimal CustomSurveyTable object with given noise data."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0, 45.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0]),
        "filter": np.array(["r", "g", "r", "g", "r", "g"]),
        "noise": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    }
    pdf = pd.DataFrame(values)
    inds = np.arange(6)

    # Try the survey with a constant flux error for all bands.
    survey_data = CustomSurveyTable(pdf, fluxerr=0.1)
    assert np.allclose(survey_data.fluxerr, np.full(6, 0.1))
    band_flux_err = survey_data.bandflux_error_point_source(np.ones(6), inds)
    assert np.allclose(band_flux_err, np.full(6, 0.1))

    # We fail with a negative constant error.
    with pytest.raises(ValueError):
        _ = CustomSurveyTable(pdf, fluxerr=-0.1)

    # Try the survey with a list of flux errors.
    survey_data = CustomSurveyTable(pdf, fluxerr=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    assert np.allclose(survey_data.fluxerr, np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2]))
    band_flux_err = survey_data.bandflux_error_point_source(np.ones(6), inds)
    assert np.allclose(band_flux_err, np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2]))

    # We fail if the list is the wrong size.
    with pytest.raises(ValueError):
        _ = CustomSurveyTable(pdf, fluxerr=[0.2, 0.4, 0.6])

    # Try the survey with a column name for the flux errors.
    survey_data = CustomSurveyTable(pdf, fluxerr="noise")
    assert np.allclose(survey_data.fluxerr, values["noise"])
    band_flux_err = survey_data.bandflux_error_point_source(np.ones(6), inds)
    assert np.allclose(band_flux_err, values["noise"])

    # We fail if the column name does not exist.
    with pytest.raises(ValueError):
        _ = CustomSurveyTable(pdf, fluxerr="something_random")

    # Try with an np array of flux errors.
    survey_data = CustomSurveyTable(pdf, fluxerr=np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8]))
    assert np.allclose(survey_data.fluxerr, np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8]))
    band_flux_err = survey_data.bandflux_error_point_source(np.ones(6), inds)
    assert np.allclose(band_flux_err, np.array([0.3, 0.6, 0.9, 1.2, 1.5, 1.8]))

    # We can use a limited set of indices.
    limited_inds = np.array([0, 2, 4])
    band_flux_err = survey_data.bandflux_error_point_source(np.ones(6), limited_inds)
    assert np.allclose(band_flux_err, np.array([0.3, 0.9, 1.5]))

    # Try with None (zero error for each point).
    survey_data = CustomSurveyTable(pdf, fluxerr=None)
    assert np.allclose(survey_data.fluxerr, np.zeros(6))
    band_flux_err = survey_data.bandflux_error_point_source(np.ones(6), inds)
    assert np.allclose(band_flux_err, np.zeros(6))

    # Try with a dictionary of constant errors per filter.
    survey_data = CustomSurveyTable(pdf, fluxerr={"r": 0.1, "g": 0.2})
    assert np.allclose(survey_data.fluxerr, [0.1, 0.2, 0.1, 0.2, 0.1, 0.2])
    band_flux_err = survey_data.bandflux_error_point_source(np.ones(6), inds)
    assert np.allclose(band_flux_err, [0.1, 0.2, 0.1, 0.2, 0.1, 0.2])

    # We fail if a filter in the table is not in the dictionary or the
    # dictionary has invalid values.
    with pytest.raises(ValueError):
        _ = CustomSurveyTable(pdf, fluxerr={"r": 0.1, "i": 0.2})
    with pytest.raises(TypeError):
        _ = CustomSurveyTable(pdf, fluxerr={"r": 0.1, "g": [0.2, 0.3]})
    with pytest.raises(ValueError):
        _ = CustomSurveyTable(pdf, fluxerr={"r": -0.1, "g": 0.2})


def _calling_fun(bandflux, table):
    """Compute the noise from the a given bandfluxes and table.

    Parameters
    ----------
    bandflux : array_like of float
        The band bandflux of the point source in nJy.
    table : pandas.core.frame.DataFrame
        The table with all the ObsTable information.

    Returns
    -------
    flux_err : array_like of float
        Simulated bandflux noise in nJy.
    """
    # Simple example: noise is 10% of the flux plus whatever is in the "noise" column.
    return 0.1 * bandflux + table["noise"].values


def test_create_custom_survey_table_callable():
    """Create a minimal CustomSurveyTable object with given noise data."""
    values = {
        "time": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0, 45.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0]),
        "filter": np.array(["r", "g", "r", "g", "r", "g"]),
        "noise": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    }
    pdf = pd.DataFrame(values)

    # Try the survey with a callable function.
    survey_data = CustomSurveyTable(pdf, fluxerr=_calling_fun)
    band_flux_err = survey_data.bandflux_error_point_source(
        np.array([10.0, 20.0, 10.0, 20.0, 10.0, 20.0]),
        np.arange(6),
    )
    assert np.allclose(band_flux_err, [1.1, 2.2, 1.3, 2.4, 1.5, 2.6])

    # Use the same callable function with a subset of indices.
    inds = np.array([0, 2, 3, 5])
    band_flux_err = survey_data.bandflux_error_point_source(
        np.array([10.0, 20.0, 10.0, 20.0]),
        inds,
    )
    assert np.allclose(band_flux_err, [1.1, 2.3, 1.4, 2.6])


def test_create_custom_survey_table_invalid():
    """Create a minimal CustomSurveyTable object with given noise data."""
    values = {
        "ra": np.array([15.0, 30.0, 15.0, 0.0, 60.0, 45.0]),
        "dec": np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0]),
        "filter": np.array(["r", "g", "r", "g", "r", "g"]),
        "noise": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    }
    pdf = pd.DataFrame(values)

    # Time column is missing.
    with pytest.raises(KeyError):
        _ = CustomSurveyTable(pdf, fluxerr=1.0)

    values["time"] = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    # Invalid fluxerr option (no support for pandas dataframe).
    with pytest.raises(TypeError):
        _ = CustomSurveyTable(pd.DataFrame(values), fluxerr=pdf)
