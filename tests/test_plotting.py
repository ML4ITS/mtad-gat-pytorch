"""Tests of the figures of the Plotter.

The tests check the structure of a figure and a checksum of its data. They do not compare the
full JSON of a figure, because a new version of plotly would change it.

Some tests hold a defect of the past. The comment names the defect.
"""

import numpy as np
import pytest

from plotting import Plotter, get_y_height


def y_checksum(trace):
    y = np.asarray(trace.y, dtype=float)
    return len(y), round(float(np.nanmin(y)), 6), round(float(np.nanmax(y)), 6), round(float(np.nansum(y)), 3)


class TestPlotFeature:
    def test_it_returns_one_figure_for_each_series(self, smd_result):
        plotter = Plotter(str(smd_result), model_id="")

        figures = plotter.plot_feature(feature=0, plot_train=True, plot_errors=True, start=0, end=500)

        assert len(figures) == 4  # test values, test error, train values, train error
        assert [len(f.data) for f in figures] == [3, 1, 3, 1]

    def test_the_data_of_the_figure_is_stable(self, smd_result):
        plotter = Plotter(str(smd_result), model_id="")

        figures = plotter.plot_feature(feature=0, plot_train=False, plot_errors=False, start=0, end=500)

        names = [t.name for t in figures[0].data]
        assert names == ["y_true", "y_forecast", "y_recon"]
        assert y_checksum(figures[0].data[0]) == (500, 0.02174, 0.52174, 88.739)
        assert y_checksum(figures[0].data[1]) == (500, 0.042389, 0.408311, 79.19)
        assert y_checksum(figures[0].data[2]) == (500, 0.039595, 0.427443, 87.029)

    def test_it_fails_for_a_feature_that_is_not_present(self, smd_result):
        plotter = Plotter(str(smd_result), model_id="")

        with pytest.raises(Exception, match="not present"):
            plotter.plot_feature(feature=999)


class TestPlotAnomalySegments:
    def test_the_height_follows_the_number_of_rows(self, smd_result, msl_result):
        """Defect of the past: the height was always 1800 pixels. MSL has one feature only,
        thus its one row was 1800 pixels high."""
        smd = Plotter(str(smd_result), model_id="").plot_anomaly_segments(split="test")
        msl = Plotter(str(msl_result), model_id="").plot_anomaly_segments(split="test")

        assert len(smd.data) == 32
        assert len(msl.data) == 1
        assert smd.layout.height > msl.layout.height
        assert msl.layout.height == 400

    def test_every_row_has_a_label(self, smd_result):
        fig = Plotter(str(smd_result), model_id="").plot_anomaly_segments(split="test")

        assert len(fig.layout.annotations) == len(fig.data)

    def test_the_labels_use_paper_coordinates(self, smd_result):
        """Defect of the past: a label moved 523 pixels to the left with the property xshift.
        Plotly then made the x axis two times as wide as the data."""
        fig = Plotter(str(smd_result), model_id="").plot_anomaly_segments(split="test")

        for annotation in fig.layout.annotations:
            assert annotation.xref == "paper"
            assert annotation.xshift in (None, 0)


class TestFigureWidth:
    """Defect of the past: a fixed width made the figure go over the edge of the notebook."""

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_no_figure_has_a_fixed_width(self, smd_result, split):
        plotter = Plotter(str(smd_result), model_id="")

        figures = [
            *plotter.plot_feature(feature=0, plot_train=True, start=0, end=500),
            plotter.plotly_global_predictions(split=split),
            plotter.plot_anomaly_segments(split=split),
        ]

        for fig in figures:
            assert fig.layout.width is None


class TestGetYHeight:
    def test_one_large_value_does_not_compress_the_series(self):
        """Defect of the past: the top of the y axis was the maximum value. The MSL test data
        holds a value of 46, thus all the other values were a flat line."""
        y = np.concatenate([np.random.default_rng(0).uniform(0, 1, 10_000), [46.0]])

        assert get_y_height(y) < 1.5

    def test_it_keeps_the_scale_of_normalized_data(self):
        y = np.linspace(0, 1, 1000)

        assert get_y_height(y) == pytest.approx(1.1, rel=1e-3)

    def test_a_series_of_zeros_gets_a_small_height(self):
        assert get_y_height(np.zeros(100)) == pytest.approx(0.1)


class TestGetAnomalySequences:
    def test_it_finds_the_start_and_the_end_of_each_sequence(self):
        values = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])

        assert Plotter.get_anomaly_sequences(values) == [[2, 4], [7, 8]]

    def test_it_finds_a_sequence_at_the_start(self):
        values = np.array([1, 1, 0, 0])

        assert Plotter.get_anomaly_sequences(values) == [[0, 1]]

    def test_it_finds_a_sequence_at_the_end(self):
        values = np.array([0, 0, 1, 1])

        assert Plotter.get_anomaly_sequences(values) == [[2, 3]]

    def test_it_finds_nothing_in_a_series_without_anomalies(self):
        assert Plotter.get_anomaly_sequences(np.zeros(10, dtype=int)) == []
