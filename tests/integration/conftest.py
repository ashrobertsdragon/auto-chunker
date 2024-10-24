from pathlib import Path

import pytest


@pytest.fixture
def book() -> str:
    with Path("tests", "integration", "sample", "test_sample.txt").open(
        "r"
    ) as f:
        return f.read()
