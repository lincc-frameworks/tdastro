import numpy as np
import pandas as pd
import pytest
from lightcurvelynx.astro_utils.mag_flux import flux2mag
from lightcurvelynx.utils.post_process_results import (
    results_augment_lightcurves,
    results_drop_empty,
)
from nested_pandas import NestedFrame


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


def test_results_augment_lightcurves():
    """Test the results_augment_lightcurves function."""
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
        results_augment_lightcurves(results, min_snr=5)

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
    assert "detection" not in results["lightcurve"].nest.fields
    assert "mag" not in results["lightcurve"].nest.fields
    assert "magerr" not in results["lightcurve"].nest.fields
    assert results["object_id"].tolist() == [0, 1, 2]

    # Augmenting the lightcurves should add the new columns.
    aug_results = results_augment_lightcurves(results, min_snr=5)
    assert len(aug_results) == 3

    # Check the SNR and detection markings.
    assert "snr" in aug_results["lightcurve"].nest.fields
    assert np.allclose(aug_results["lightcurve.snr"][0].values, [10.0, 12.0])
    assert np.allclose(aug_results["lightcurve.snr"][1].values, [0.1, 0.01])
    assert np.allclose(aug_results["lightcurve.snr"][2].values, [2.5, 6.0])

    assert "detection" in aug_results["lightcurve"].nest.fields
    assert aug_results["lightcurve.detection"][0].tolist() == [True, True]
    assert aug_results["lightcurve.detection"][1].tolist() == [False, False]
    assert aug_results["lightcurve.detection"][2].tolist() == [False, True]

    # Check the AB magnitudes and magnitude errors.
    assert "mag" in aug_results["lightcurve"].nest.fields
    assert np.allclose(aug_results["lightcurve.mag"][0].values, [flux2mag(10.0), flux2mag(12.0)])
    assert np.allclose(aug_results["lightcurve.mag"][1].values, [flux2mag(0.1), flux2mag(0.2)])
    assert np.allclose(aug_results["lightcurve.mag"][2].values, [flux2mag(5.0), flux2mag(6.0)])

    assert "magerr" in aug_results["lightcurve"].nest.fields
    for i in range(3):
        assert np.allclose(
            aug_results["lightcurve.magerr"][i].values,
            (2.5 / np.log(10)) * 1.0 / aug_results["lightcurve.snr"][i].values,
        )


def test_results_augment_lightcurves_single():
    """Test the results_augment_lightcurves function with a non-nested frame."""
    # Create a DataFrame with a lightcurve.
    source_data = {
        "mjd": [59000, 59001, 59002, 59003, 59004, 59005],
        "flux": [10.0, 12.0, 0.1, 0.2, 5.0, 6.0],
    }
    results = pd.DataFrame(data=source_data)

    with pytest.raises(ValueError):
        # Ensure that the function raises an error if 'flux' or 'fluxerr' is not present.
        results_augment_lightcurves(results, min_snr=5)

    # Create a nested DataFrame with lightcurves, some of which are empty.
    results["fluxerr"] = [1.0, 1.0, 1.0, 20.0, 2.0, 1.0]

    # Add the nested DataFrame to the results.
    assert "snr" not in results.columns
    assert "detection" not in results.columns
    assert "mag" not in results.columns
    assert "magerr" not in results.columns

    # Augmenting the lightcurves should add the new columns.
    results = results_augment_lightcurves(results, min_snr=5)

    # Check we have added the columns.
    assert "snr" in results.columns
    assert "detection" in results.columns
    assert "mag" in results.columns
    assert "magerr" in results.columns
    assert np.allclose(results["snr"].values, [10.0, 12.0, 0.1, 0.01, 2.5, 6.0])
    assert np.array_equal(results["detection"].values, [True, True, False, False, False, True])
    assert np.allclose(
        results["mag"].values,
        [flux2mag(10.0), flux2mag(12.0), flux2mag(0.1), flux2mag(0.2), flux2mag(5.0), flux2mag(6.0)],
    )
    assert np.allclose(results["magerr"].values, (2.5 / np.log(10)) * (1.0 / results["snr"].values))
