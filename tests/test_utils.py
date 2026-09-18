"""Tests of the helper functions in utils.py."""

import numpy as np
import pytest
import torch

from dataset_info import get_data_dim, get_target_dims
from utils import adjust_anomaly_scores, get_device


class TestGetDevice:
    def test_it_gives_the_cpu_if_the_user_does_not_want_a_gpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        assert get_device(use_gpu=False) == "cpu"

    def test_it_gives_cuda_first(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        assert get_device() == "cuda"

    def test_it_gives_mps_if_the_machine_has_no_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        assert get_device() == "mps"

    def test_it_gives_the_cpu_if_the_machine_has_no_gpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        assert get_device() == "cpu"

    def test_the_device_works(self):
        """The name must be a device that torch accepts."""
        torch.zeros(1).to(get_device())


class TestAdjustAnomalyScores:
    """The function reads the metadata CSV files. Those files are in the repository."""

    def test_it_returns_the_scores_of_smd_without_a_change(self):
        scores = np.arange(10, dtype=float)

        result = adjust_anomaly_scores(scores, "SMD", is_train=False, lookback=100)

        assert np.array_equal(result, scores)

    # The function needs one score for each time step of all the channels together.
    # The number is the sum of the column num_values of the metadata, minus the lookback.
    LENGTHS = {("MSL", True): 58_217, ("MSL", False): 73_629, ("SMAP", True): 135_083, ("SMAP", False): 427_517}

    @pytest.mark.parametrize("dataset", ["MSL", "SMAP"])
    @pytest.mark.parametrize("is_train", [True, False])
    def test_it_normalizes_each_channel_of_msl_and_smap(self, dataset, is_train):
        rng = np.random.default_rng(0)
        scores = rng.uniform(1.0, 2.0, self.LENGTHS[(dataset, is_train)])

        result = adjust_anomaly_scores(scores.copy(), dataset, is_train=is_train, lookback=100)

        assert result.shape == scores.shape
        assert not np.array_equal(result, scores)
        # A channel is scaled to the range 0 to 1, and the border between two channels is 0.
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    @pytest.mark.parametrize("length", [1000, 58_216])
    def test_it_accepts_scores_that_stop_before_the_last_channel(self, length):
        """The user can limit the size of the data with max_test_size. The scores then stop
        in the middle of the channels, and the last parts are empty."""
        scores = np.random.default_rng(2).uniform(1.0, 2.0, length)

        result = adjust_anomaly_scores(scores.copy(), "MSL", is_train=True, lookback=100)

        assert result.shape == scores.shape
        assert np.isfinite(result).all()

    def test_it_accepts_a_constant_series(self):
        scores = np.ones(self.LENGTHS[("MSL", True)])

        result = adjust_anomaly_scores(scores.copy(), "MSL", is_train=True, lookback=100)

        assert np.isfinite(result).all()

    def test_it_does_not_change_the_input(self):
        scores = np.random.default_rng(1).uniform(1.0, 2.0, self.LENGTHS[("MSL", True)])
        before = scores.copy()

        adjust_anomaly_scores(scores, "MSL", is_train=True, lookback=100)

        assert np.array_equal(scores, before)


class TestDatasetInfo:
    @pytest.mark.parametrize(("dataset", "dim"), [("SMAP", 25), ("MSL", 55), ("machine-1-1", 38)])
    def test_it_gives_the_number_of_features(self, dataset, dim):
        assert get_data_dim(dataset) == dim

    @pytest.mark.parametrize(("dataset", "dims"), [("SMAP", [0]), ("MSL", [0]), ("SMD", None)])
    def test_it_gives_the_dimensions_to_forecast(self, dataset, dims):
        assert get_target_dims(dataset) == dims

    def test_it_fails_for_an_unknown_dataset(self):
        with pytest.raises(ValueError, match="unknown dataset"):
            get_data_dim("NOT_A_DATASET")
