from pathlib import Path

# Define some global directory paths to use for testing, notebooks, etc.
_TDASTRO_BASE_DIR = Path(__file__).parent.parent.parent
_TDASTRO_BASE_DATA_DIR = _TDASTRO_BASE_DIR / "data"

_TDASTRO_TEST_DIR = _TDASTRO_BASE_DIR / "tests" / "tdastro"
_TDASTRO_TEST_DATA_DIR = _TDASTRO_TEST_DIR / "data"
