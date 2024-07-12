import os.path

import pytest

DATA_DIR_NAME = "data"
TEST_DIR = os.path.dirname(__file__)


@pytest.fixture
def test_data_dir():
    """Return the base test data directory."""
    return os.path.join(TEST_DIR, DATA_DIR_NAME)


@pytest.fixture
def grid_data_good_file(test_data_dir):
    """Return the file path for the good grid input file."""
    return os.path.join(test_data_dir, "grid_input_good.ecsv")


@pytest.fixture
def grid_data_bad_file(test_data_dir):
    """Return the file path for the bad grid input file."""
    return os.path.join(test_data_dir, "grid_input_bad.txt")
