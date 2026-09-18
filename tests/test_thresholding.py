"""Tests of the POT threshold."""

import numpy as np
import polars as pl
import pytest

from thresholding import pot_threshold

# The values come from the code before the module thresholding.py. The parameters of SMD group 1
# are in predict.py.
SMD_Q, SMD_LEVEL = 0.001, 0.9950
SMD_THRESHOLD = 0.1366700058144852


@pytest.fixture
def smd_scores(smd_result):
    train = pl.read_parquet(smd_result / "train_output.parquet")["A_Score_Global"].to_numpy()
    test = pl.read_parquet(smd_result / "test_output.parquet")["A_Score_Global"].to_numpy()
    return train, test


def test_it_gives_the_threshold_of_the_earlier_code(smd_scores):
    train, test = smd_scores

    threshold = pot_threshold(train, test, q=SMD_Q, level=SMD_LEVEL)

    assert threshold == pytest.approx(SMD_THRESHOLD, rel=1e-12)


def test_the_test_scores_do_not_change_the_result(smd_scores):
    """With dynamic=False the method uses the train scores only."""
    train, test = smd_scores

    with_stream = pot_threshold(train, test, q=SMD_Q, level=SMD_LEVEL)
    without_stream = pot_threshold(train, q=SMD_Q, level=SMD_LEVEL)

    assert with_stream == without_stream


def test_the_dynamic_method_gives_another_threshold(smd_scores):
    """With dynamic=True the threshold follows the test scores."""
    train, test = smd_scores

    static = pot_threshold(train, test, q=SMD_Q, level=SMD_LEVEL, dynamic=False)
    dynamic = pot_threshold(train, test, q=SMD_Q, level=SMD_LEVEL, dynamic=True)

    assert dynamic != static
    assert dynamic > 0


def test_a_larger_risk_gives_a_lower_threshold(smd_scores):
    train, _ = smd_scores

    low_risk = pot_threshold(train, q=1e-4, level=SMD_LEVEL)
    high_risk = pot_threshold(train, q=1e-2, level=SMD_LEVEL)

    assert high_risk < low_risk


def test_the_type_of_the_scores_stays(smd_scores):
    """A cast from float32 to float64 moves the threshold, thus the function must not cast."""
    train, _ = smd_scores

    assert train.dtype == np.float32
    assert pot_threshold(train, q=SMD_Q, level=SMD_LEVEL) == pytest.approx(SMD_THRESHOLD, rel=1e-12)
    assert pot_threshold(train.astype(np.float64), q=SMD_Q, level=SMD_LEVEL) != pytest.approx(SMD_THRESHOLD, rel=1e-12)
