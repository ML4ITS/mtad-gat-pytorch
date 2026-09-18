# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "plotly",
#     "polars",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from plotting import Plotter

    return Path, Plotter, json, mo, os, pl


@app.cell
def _(mo):
    mo.md("""
    # Result Visualizer

    Different visualizations of anomaly detection results: forecasts, reconstructions,
    anomaly scores, predicted and actual anomalies.

    Predicted anomalies are shown with a **blue** rectangle, actual (true) anomalies with a
    **red** one — so a correctly predicted anomaly shows up **purple**.
    """)
    return


@app.cell
def _(Path, mo):
    result_dirs = sorted({str(p.parent) for p in Path("output").rglob("test_output.parquet")})

    result_dir = mo.ui.dropdown(
        options=result_dirs,
        value=result_dirs[0] if result_dirs else None,
        label="Result directory",
    )
    result_dir
    return (result_dir,)


@app.cell
def _(Plotter, mo, result_dir):
    mo.stop(
        result_dir.value is None,
        mo.md("**No results found.** Train a model first, or check that `output/` holds `test_output.parquet`."),
    )

    plotter = Plotter(result_dir.value, model_id="")
    return (plotter,)


@app.cell
def _(json, mo, os, pl, plotter):
    summary_path = f"{plotter.result_path}/summary.txt"

    method_names = {"epsilon_result": "epsilon", "pot_result": "peaks-over-threshold", "bf_result": "brute-force"}

    summary = (
        pl.DataFrame(
            [
                {
                    "method": method_names.get(method, method),
                    **{k: v for k, v in res.items() if k in ("precision", "recall", "f1")},
                }
                for method, res in json.load(open(summary_path)).items()
                if isinstance(res, dict) and res
            ]
        )
        if os.path.exists(summary_path)
        else pl.DataFrame()
    )

    mo.vstack([mo.md(f"### Performance on test set — `{plotter.result_path}`"), summary])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interactive feature plot

    Forecasts, reconstructions, actual values and anomaly scores for a single feature.
    With *feature-level anomalies* enabled, the feature-wise threshold and its anomaly
    predictions are drawn as well.
    """)
    return


@app.cell
def _(mo, plotter):
    n_features = len(plotter.pred_cols) if plotter.pred_cols is not None else 1
    n_rows = plotter.test_output.height

    feature = mo.ui.slider(0, max(n_features - 1, 0), 1, value=0, label="Feature", show_value=True)
    index_range = mo.ui.range_slider(
        0, n_rows, step=max(n_rows // 500, 1), value=[0, min(3000, n_rows)], label="Index range", show_value=True
    )
    plot_train = mo.ui.checkbox(value=True, label="Also plot train set")
    plot_errors = mo.ui.checkbox(value=True, label="Plot anomaly scores")
    plot_feature_anom = mo.ui.checkbox(value=True, label="Feature-level anomalies")

    mo.vstack([feature, index_range, mo.hstack([plot_train, plot_errors, plot_feature_anom], justify="start")])
    return feature, index_range, plot_errors, plot_feature_anom, plot_train


@app.cell
def _(
    feature,
    index_range,
    mo,
    plot_errors,
    plot_feature_anom,
    plot_train,
    plotter,
):
    feature_figures = plotter.plot_feature(
        feature=feature.value,
        plot_train=plot_train.value,
        plot_errors=plot_errors.value,
        plot_feature_anom=plot_feature_anom.value,
        start=index_range.value[0],
        end=index_range.value[1],
    )

    mo.vstack([mo.ui.plotly(fig) for fig in feature_figures])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Global (entity-level) anomaly predictions

    Entity-level anomaly scores, which are what the anomaly predictions are made from.
    """)
    return


@app.cell
def _(mo):
    split = mo.ui.radio(options=["test", "train"], value="test", label="Split", inline=True)
    split
    return (split,)


@app.cell
def _(mo, plotter, split):
    mo.ui.plotly(plotter.plotly_global_predictions(split=split.value))
    return


@app.cell
def _(plotter, split):
    plotter.plot_global_predictions(split=split.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Collective anomalies

    Feature-wise anomalies that occur at the same time. Only meaningful for multivariate
    output (SMD) — it is slow for wide datasets, so it is off by default.
    """)
    return


@app.cell
def _(mo):
    show_segments = mo.ui.checkbox(value=False, label="Plot anomaly segments")
    show_segments
    return (show_segments,)


@app.cell
def _(mo, plotter, show_segments, split):
    mo.stop(not show_segments.value, mo.md("*Enable the checkbox above to render the segment plot.*"))

    mo.ui.plotly(plotter.plot_anomaly_segments(split=split.value, num_aligned_segments=None))
    return


if __name__ == "__main__":
    app.run()
