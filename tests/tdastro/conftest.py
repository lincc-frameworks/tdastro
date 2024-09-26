import os.path

import pytest

DATA_DIR_NAME = "data"
TEST_DIR = os.path.dirname(__file__)


def fixture(func):
    """Decorator to make a function a fixture.

    We use to make it to use fixtures as regular functions when we have to.
    """
    func_name = func.__name__
    fixture_func_name = f"{func_name}_fixture"

    fixture_func = pytest.fixture(func, name=func_name)
    fixture_func.__name__ = fixture_func_name
    globals()[fixture_func_name] = fixture_func

    return func


@fixture
def test_data_dir():
    """Return the base test data directory."""
    return os.path.join(TEST_DIR, DATA_DIR_NAME)


@fixture
def grid_data_good_file(test_data_dir):
    """Return the file path for the good grid input file."""
    return os.path.join(test_data_dir, "grid_input_good.ecsv")


@fixture
def grid_data_bad_file(test_data_dir):
    """Return the file path for the bad grid input file."""
    return os.path.join(test_data_dir, "grid_input_bad.txt")


@fixture
def opsim_small(test_data_dir):
    """Return the file path for the bad grid input file."""
    return os.path.join(test_data_dir, "opsim_small.db")


@fixture
def opsim_shorten(test_data_dir):
    """Return the file path for the bad grid input file."""
    return os.path.join(test_data_dir, "opsim_shorten.db")


@fixture
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


@fixture
def passbands_dir(test_data_dir):
    """Return the file path for passbands directory."""
    return os.path.join(test_data_dir, "passbands")
