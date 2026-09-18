"""Shared fixtures.

The tests run against the example results committed in `output/`. Those are parquet files,
so no dataset download and no training is needed.
"""

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

EXAMPLE_RESULTS = {
    "SMD": REPO_ROOT / "output" / "SMD" / "1-1" / "27062021_114402",
    "MSL": REPO_ROOT / "output" / "MSL" / "27062021_111641",
}


def _copy_result(dataset, tmp_path):
    """Copy an example result into a temporary directory.

    The Plotter and the Predictor both write into the result directory, so a test must never
    use the copy in the repository. The name of the dataset stays in the path, because the
    Plotter reads the path to find the number of features.
    """
    source = EXAMPLE_RESULTS[dataset]
    target = tmp_path / dataset / "run"
    target.mkdir(parents=True)
    for name in ["train_output.parquet", "test_output.parquet", "config.txt", "summary.txt"]:
        shutil.copy(source / name, target / name)
    return target


@pytest.fixture
def smd_result(tmp_path):
    return _copy_result("SMD", tmp_path)


@pytest.fixture
def msl_result(tmp_path):
    return _copy_result("MSL", tmp_path)
