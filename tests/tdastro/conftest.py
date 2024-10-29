from pathlib import Path

import pytest
from tdastro import _TDASTRO_TEST_DATA_DIR


@pytest.fixture
def test_data_dir():
    """Return the base test data directory."""
    return Path(_TDASTRO_TEST_DATA_DIR)


@pytest.fixture
def grid_data_good_file(test_data_dir):
    """Return the file path for the good grid input file."""
    return test_data_dir / "grid_input_good.ecsv"


@pytest.fixture
def grid_data_bad_file(test_data_dir):
    """Return the file path for the bad grid input file."""
    return test_data_dir / "grid_input_bad.txt"


@pytest.fixture
def opsim_small(test_data_dir):
    """Return the file path for the bad grid input file."""
    return test_data_dir / "opsim_small.db"


@pytest.fixture
def opsim_shorten(test_data_dir):
    """Return the file path for the bad grid input file."""
    return test_data_dir / "opsim_shorten.db"


@pytest.fixture
def oversampled_observations(opsim_shorten):
    """Return an OpSim object with 0.01 day cadence spanning year 2027."""
    from tdastro.astro_utils.opsim import OpSim, oversample_opsim

    base_opsim = OpSim.from_db(opsim_shorten)
    return oversample_opsim(
        base_opsim,
        pointing=(0.0, 0.0),
        search_radius=180.0,
        delta_t=0.01,
        time_range=(61406.0, 61771.0),
        bands=None,
        strategy="darkest_sky",
    )


@pytest.fixture
def passbands_dir(test_data_dir):
    """Return the file path for passbands directory."""
    return test_data_dir / "passbands"
