"""Fixtures for testing TDAstro"""

import os
import pytest
from pathlib import Path

TEST_DIR = Path(__file__)
TOP_LEVEL_DIR = TEST_DIR.parent.parent.parent


@pytest.fixture
def tdastro_data_dir():
    return os.path.join(TOP_LEVEL_DIR, "data")
