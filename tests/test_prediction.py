"""Regression test of the metrics of the prediction pipeline.

The Predictor reads the saved anomaly scores when load_scores is True. It does not use the
model or the train and test data then. Thus this test covers prediction.py, eval_methods.py
and spot.py without a model and without the datasets.

The expected values come from the pandas code of the repository before the polars change.
"""

import json

import polars as pl
import pytest

from prediction import Predictor

# SMD group 1 uses these values. See predict.py.
SMD_PRED_ARGS = {
    "dataset": "SMD",
    "target_dims": None,
    "scale_scores": False,
    "level": 0.9950,
    "q": 0.001,
    "dynamic_pot": False,
    "use_mov_av": False,
    "gamma": 1,
    "reg_level": 1,
    "use_cuda": False,
}

EXPECTED = {
    "epsilon_result": {"f1": 0.861110647816744, "precision": 0.7561044042770013, "recall": 0.9999999962880475},
    "pot_result": {"f1": 0.7031143264500973, "precision": 0.5421613995931548, "recall": 0.9999999962880475},
    "bf_result": {"f1": 0.9998094442891704, "precision": 0.999628938776887, "recall": 0.9999999962880475},
}


def run_predictor(result_dir, **kwargs):
    """Run the predictor on saved scores and return the summary it writes."""
    labels = pl.read_parquet(result_dir / "test_output.parquet")["A_True_Global"].to_numpy()
    predictor = Predictor(
        model=None,
        window_size=100,
        n_features=38,
        pred_args={**SMD_PRED_ARGS, "save_path": str(result_dir), **kwargs},
    )
    predictor.predict_anomalies(None, None, labels, load_scores=True, save_output=False)
    with open(result_dir / "summary.txt") as f:
        return json.load(f)


def test_metrics_of_the_three_methods(smd_result):
    summary = run_predictor(smd_result)

    for method, expected in EXPECTED.items():
        for name, value in expected.items():
            assert summary[method][name] == pytest.approx(value, rel=1e-6), f"{method}.{name}"


def test_metrics_agree_with_the_original_summary(smd_result):
    """The summary of the training run of 2021 comes from the pandas code."""
    with open(smd_result / "summary.txt") as f:
        original = json.load(f)

    summary = run_predictor(smd_result)

    for method in EXPECTED:
        for name in ["f1", "precision", "recall", "TP", "TN", "FP", "FN"]:
            assert summary[method][name] == pytest.approx(original[method][name], rel=1e-6), f"{method}.{name}"


def test_the_predictor_writes_the_results(smd_result):
    labels = pl.read_parquet(smd_result / "test_output.parquet")["A_True_Global"].to_numpy()
    predictor = Predictor(
        model=None,
        window_size=100,
        n_features=38,
        pred_args={**SMD_PRED_ARGS, "save_path": str(smd_result)},
    )
    predictor.predict_anomalies(None, None, labels, load_scores=True, save_output=True)

    test_output = pl.read_parquet(smd_result / "test_output.parquet")
    for column in ["A_Score_Global", "A_Pred_Global", "Thresh_Global", "A_True_Global"]:
        assert column in test_output.columns
    # The last four columns hold the global results. The order of the columns is important,
    # because plot_anomaly_segments reads the columns of a feature by their position.
    assert test_output.columns[:4] == ["Forecast_0", "Recon_0", "True_0", "A_Score_0"]
