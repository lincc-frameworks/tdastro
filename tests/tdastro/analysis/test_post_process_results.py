import pandas as pd
import pytest
from nested_pandas import NestedFrame
from tdastro.analysis.post_process_results import (
    results_append_lightcurve_dt,
    results_append_lightcurve_snr,
    results_append_num_filters,
    results_append_num_obs,
    results_drop_empty,
)


def test_results_drop_empty():
    """Test the results_drop_empty function."""
    # Create a NestedFrame with some empty lightcurves.
    source_data = {
        "object_id": [0, 1, 2, 3, 4],
        "ra": [10.0, 20.0, 30.0, 40.0, 50.0],
        "dec": [-10.0, -20.0, -30.0, -40.0, -50.0],
        "nobs": [3, 0, 1, 2, 0],
        "z": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    results = NestedFrame(data=source_data, index=[0, 1, 2, 3, 4])

    # Create a nested DataFrame with lightcurves, some of which are empty.
    nested_data = {
        "mjd": [59000, 59001, 59002, 59003, 59004, 59005],
        "flux": [10.0, 12.0, 11.0, 20.0, 22.0, 30.0],
        "fluxerr": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
        "filter": ["g", "r", "i", "g", "r", "i"],
    }
    nested_frame = pd.DataFrame(data=nested_data, index=[0, 0, 0, 2, 3, 3])

    # Add the nested DataFrame to the results.
    results = results.add_nested(nested_frame, "lightcurve")
    assert len(results) == 5
    assert results["object_id"].tolist() == [0, 1, 2, 3, 4]

    # Apply the drop_empty function.
    filtered_results = results_drop_empty(results)
    assert len(filtered_results) == 3
    assert filtered_results["object_id"].tolist() == [0, 2, 3]

    assert len(filtered_results["lightcurve"][0]) == 3
    assert len(filtered_results["lightcurve"][2]) == 1
    assert len(filtered_results["lightcurve"][3]) == 2


def test_results_results_append_num_obs():
    """Test the results_append_num_obs function."""
    # Create with different number of observations.
    source_data = {
        "object_id": [0, 1, 2, 3, 4],
        "ra": [10.0, 20.0, 30.0, 40.0, 50.0],
        "dec": [-10.0, -20.0, -30.0, -40.0, -50.0],
        "nobs": [3, 0, 1, 2, 0],
        "z": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    results = NestedFrame(data=source_data, index=[0, 1, 2, 3, 4])

    nested_data = {
        "mjd": [
            59000,  # id=0
            59001,  # id=0
            59002,  # id=0
            59000,  # id=1 (filter)
            59000,  # id=2
            59001,  # id=2
            59000,  # id=3
            59000,  # id=3 (duplicate time)
            59005,  # id=4
            59006,  # id=4
        ],
    }
    nested_frame = pd.DataFrame(data=nested_data, index=[0, 0, 0, 1, 2, 2, 3, 3, 4, 4])
    results = results.add_nested(nested_frame, "lightcurve")

    assert len(results) == 5
    assert results["object_id"].tolist() == [0, 1, 2, 3, 4]
    assert "num_obs" not in results.columns

    # Append the number of observations.
    results = results_append_num_obs(results, min_obs=0)
    assert "num_obs" in results.columns
    assert results["num_obs"].tolist() == [3, 1, 2, 1, 2]

    # Check that the filtering works.
    filtered_results = results_append_num_obs(results, min_obs=2)
    assert len(filtered_results) == 3
    assert filtered_results["object_id"].tolist() == [0, 2, 4]


def test_results_results_append_num_filters():
    """Test the results_append_num_filters function."""
    # Create with different number of filters.
    source_data = {
        "object_id": [0, 1, 2, 3, 4],
        "ra": [10.0, 20.0, 30.0, 40.0, 50.0],
        "dec": [-10.0, -20.0, -30.0, -40.0, -50.0],
        "nobs": [3, 0, 1, 2, 0],
        "z": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    results = NestedFrame(data=source_data, index=[0, 1, 2, 3, 4])

    nested_data = {
        "filter": [
            "r",  # id=0
            "g",  # id=0
            "i",  # id=0
            "r",  # id=1 (filter)
            "r",  # id=2
            "g",  # id=2
            "r",  # id=3
            "r",  # id=3 (duplicate filter)
            "r",  # id=4
            "g",  # id=4
        ],
    }
    nested_frame = pd.DataFrame(data=nested_data, index=[0, 0, 0, 1, 2, 2, 3, 3, 4, 4])
    results = results.add_nested(nested_frame, "lightcurve")

    assert len(results) == 5
    assert results["object_id"].tolist() == [0, 1, 2, 3, 4]
    assert "num_filters" not in results.columns

    # Append the number of filters.
    results = results_append_num_filters(results, min_filters=0)
    assert "num_filters" in results.columns
    assert results["num_filters"].tolist() == [3, 1, 2, 1, 2]

    # Check that the filtering works.
    filtered_results = results_append_num_filters(results, min_filters=2)
    assert len(filtered_results) == 3
    assert filtered_results["object_id"].tolist() == [0, 2, 4]


def test_results_append_lightcurve_dt():
    """Test the results_append_lightcurve_dt function."""
    # Create with different number of observations.
    source_data = {
        "object_id": [0, 1, 2, 3, 4],
        "ra": [10.0, 20.0, 30.0, 40.0, 50.0],
        "dec": [-10.0, -20.0, -30.0, -40.0, -50.0],
        "nobs": [3, 0, 1, 2, 0],
        "z": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    results = NestedFrame(data=source_data, index=[0, 1, 2, 3, 4])

    nested_data = {
        "mjd": [
            59000,  # id=0
            59001,  # id=0
            59002,  # id=0
            59000,  # id=1 (filter - one observation)
            59000,  # id=2
            59001,  # id=2
            59000,  # id=3
            59000,  # id=3 (filter - dt = 0
            59005,  # id=4
            59006,  # id=4
        ],
    }
    nested_frame = pd.DataFrame(data=nested_data, index=[0, 0, 0, 1, 2, 2, 3, 3, 4, 4])
    results = results.add_nested(nested_frame, "lightcurve")

    assert len(results) == 5
    assert results["object_id"].tolist() == [0, 1, 2, 3, 4]
    assert "lightcurve_dt" not in results.columns

    # Append the number of observations.
    results = results_append_lightcurve_dt(results, min_dt=0)
    assert "lightcurve_dt" in results.columns
    assert results["lightcurve_dt"].tolist() == [2, 0, 1, 0, 1]

    # Check that the filtering works.
    filtered_results = results_append_lightcurve_dt(results, min_dt=1.0)
    assert len(filtered_results) == 3
    assert filtered_results["object_id"].tolist() == [0, 2, 4]


def test_results_append_lightcurve_snr():
    """Test the results_append_lightcurve_snr function."""
    # Create a NestedFrame with some empty lightcurves.
    source_data = {
        "object_id": [0, 1, 2],
        "ra": [10.0, 20.0, 30.0],
        "dec": [-10.0, -20.0, -30.0],
        "nobs": [3, 0, 1],
        "z": [0.1, 0.2, 0.3],
    }
    results = NestedFrame(data=source_data, index=[0, 1, 2])

    with pytest.raises(ValueError):
        # Ensure that the function raises an error if 'lightcurve' is not present.
        results_append_lightcurve_snr(results, min_snr=5)

    # Create a nested DataFrame with lightcurves, some of which are empty.
    nested_data = {
        "mjd": [59000, 59001, 59002, 59003, 59004, 59005],
        "flux": [10.0, 12.0, 0.1, 0.2, 5.0, 6.0],
        "fluxerr": [1.0, 1.0, 1.0, 20.0, 2.0, 1.0],
        "filter": ["g", "r", "i", "g", "r", "i"],
    }
    nested_frame = pd.DataFrame(data=nested_data, index=[0, 0, 1, 1, 2, 2])

    # Add the nested DataFrame to the results.
    results = results.add_nested(nested_frame, "lightcurve")
    assert len(results) == 3
    assert "snr" not in results["lightcurve"].nest.fields
    assert results["object_id"].tolist() == [0, 1, 2]

    # Filtering on the SNR adds the column and does the filtering.
    filtered_results = results_append_lightcurve_snr(results, 5)
    assert len(filtered_results) == 2
    assert "snr" in filtered_results["lightcurve"].nest.fields

    # Row 0 has both observations kept, row 1 has none, and row 2 has one observation kept.
    assert filtered_results["object_id"].tolist() == [0, 2]
    assert len(filtered_results["lightcurve"][0]) == 2
    assert len(filtered_results["lightcurve"][2]) == 1

    # Confirm that we fail if we try to pass a non-numeric SNR threshold.
    with pytest.raises(ValueError):
        results_append_lightcurve_snr(results, "invalid")
