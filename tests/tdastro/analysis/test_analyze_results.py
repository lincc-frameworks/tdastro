import pandas as pd
from nested_pandas import NestedFrame
from tdastro.analysis.analyze_results import (
    results_drop_empty,
    results_filter_on_snr,
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


def test_results_snr():
    """Test the results_compute_snr and results_filter_on_snr functions."""
    # Create a NestedFrame with some empty lightcurves.
    source_data = {
        "object_id": [0, 1, 2],
        "ra": [10.0, 20.0, 30.0],
        "dec": [-10.0, -20.0, -30.0],
        "nobs": [3, 0, 1],
        "z": [0.1, 0.2, 0.3],
    }
    results = NestedFrame(data=source_data, index=[0, 1, 2])

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
    filtered_results = results_filter_on_snr(results, 5)
    assert len(filtered_results) == 2
    assert "snr" in results["lightcurve"].nest.fields

    # Row 0 has both observations kept, row 1 has none, and row 2 has one observation kept.
    assert filtered_results["object_id"].tolist() == [0, 2]
    assert len(filtered_results["lightcurve"][0]) == 2
    assert len(filtered_results["lightcurve"][2]) == 1
