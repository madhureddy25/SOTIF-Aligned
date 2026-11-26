import pytest

from sotif_miner.config import load_config


@pytest.fixture
def cfg():
    return load_config()
