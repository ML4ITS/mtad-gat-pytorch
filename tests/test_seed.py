"""Tests of the seed of the random number generators."""

import numpy as np
import torch

from utils import SlidingWindowDataset, create_data_loaders, set_seed


def loader_indices(seed):
    """Build the data loaders with a seed and collect the windows they give."""
    set_seed(seed)
    dataset = SlidingWindowDataset(torch.arange(500, dtype=torch.float32).reshape(-1, 1), window=10)
    train_loader, val_loader, _ = create_data_loaders(dataset, batch_size=16, val_split=0.1, shuffle=True)
    return (
        [x.numpy().copy() for x, _ in train_loader],
        [x.numpy().copy() for x, _ in val_loader],
    )


def test_the_same_seed_gives_the_same_windows():
    first_train, first_val = loader_indices(42)
    second_train, second_val = loader_indices(42)

    assert all(np.array_equal(a, b) for a, b in zip(first_train, second_train))
    assert all(np.array_equal(a, b) for a, b in zip(first_val, second_val))


def test_another_seed_gives_other_windows():
    first_train, _ = loader_indices(42)
    other_train, _ = loader_indices(7)

    assert not all(np.array_equal(a, b) for a, b in zip(first_train, other_train))


def test_the_seed_none_keeps_the_random_behaviour():
    set_seed(42)
    set_seed(None)
    first = np.random.rand()

    set_seed(42)
    set_seed(None)
    second = np.random.rand()

    assert first == second  # set_seed(None) does not change the state that set_seed(42) made


def test_the_seed_reaches_torch_and_numpy():
    set_seed(3)
    values = (np.random.rand(), torch.rand(1).item())

    set_seed(3)

    assert (np.random.rand(), torch.rand(1).item()) == values
