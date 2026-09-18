"""Tests of the threshold methods."""

import numpy as np
import pytest

from eval_methods import bf_search, calc_point2point


@pytest.fixture
def scores_and_labels():
    rng = np.random.default_rng(0)
    score = rng.random(500)
    label = (score > 0.9).astype(int)
    return score, label


class TestBfSearch:
    def test_it_finds_a_threshold(self, scores_and_labels):
        score, label = scores_and_labels

        result = bf_search(score, label, start=0.01, end=2, step_num=100, verbose=False)

        assert 0.0 <= result["f1"] <= 1.0
        assert set(result) == {"f1", "precision", "recall", "TP", "TN", "FP", "FN", "threshold", "latency"}

    def test_it_accepts_a_search_with_no_step(self, scores_and_labels):
        """Defect of the past: the first value held three elements, but the function reads
        seven elements. A search with no step gave an IndexError."""
        score, label = scores_and_labels

        result = bf_search(score, label, start=0.01, end=2, step_num=0, verbose=False)

        assert result["f1"] == -1.0


class TestCalcPoint2Point:
    def test_it_counts_the_correct_and_the_false_predictions(self):
        predict = np.array([1, 1, 0, 0])
        actual = np.array([1, 0, 1, 0])

        f1, precision, recall, tp, tn, fp, fn = calc_point2point(predict, actual)

        assert (tp, tn, fp, fn) == (1, 1, 1, 1)
        # The functions add 0.00001 to each divisor, thus the values are a little under 0.5.
        assert precision == pytest.approx(0.5, abs=1e-4)
        assert recall == pytest.approx(0.5, abs=1e-4)
        assert f1 == pytest.approx(0.5, abs=1e-4)
