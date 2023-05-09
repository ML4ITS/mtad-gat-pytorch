import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, SequentialSampler

# TODO: REMOVE DATA NORMALIZATION AND ANY OTHER TYPE OF DATA PRE-PROCESSING
# ONLY THE MODEL FILES SHOULD BE HERE, NOT OTHER TASKS
def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler


def get_data(dataset, max_train_size=None, max_test_size=None, normalize=False, train_start=0, test_start=0):
    
    dataset_folder = os.path.join("datasets", dataset)

    print("Loading data for dataset:", dataset)
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size

    # Load the data
    train_data = np.loadtxt(os.path.join(dataset_folder, "train.txt"),
                            delimiter=",", dtype=np.float32)[train_start:train_end, :]
    try:
        test_data = np.loadtxt(os.path.join(dataset_folder, "test.txt"),
                                delimiter=",", dtype=np.float32)[test_start:test_end, :]
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        test_label = np.loadtxt(os.path.join(dataset_folder, "labels.txt"),
                                delimiter=",", dtype=np.float32)[test_start:test_end]
    except (KeyError, FileNotFoundError):
        test_label = None

    # TODO: REMOVE DATA NORMALIZATION AND ANY OTHER TYPE OF DATA PRE-PROCESSING
    # ONLY THE MODEL FILES SHOULD BE HERE, NOT OTHER TASKS
    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None)
        test_data, _ = normalize_data(test_data, scaler=scaler)

    return (train_data, None), (test_data, test_label)


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, horizon=1):
        self.data = data
        self.window = window
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"The size of the training dataset is: {len(train_dataset)} samples.")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        if shuffle:
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)
        else:
            train_sampler = SequentialSampler(train_indices)
            valid_sampler = SequentialSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"The size of the training dataset is: {len(train_indices)} samples.")
        print(f"The size of the validation dataset is: {len(val_indices)} samples.")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"The size of the validation dataset is: {len(test_dataset)} samples.")

    return train_loader, val_loader, test_loader


def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.plot(losses["train_recon"], label="Recon loss")
    plt.plot(losses["train_total"], label="Total loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.plot(losses["val_recon"], label="Recon loss")
    plt.plot(losses["val_total"], label="Total loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()


def load(model, PATH, device="cpu"):
    """
    Loads the model's parameters from the path mentioned
    :param PATH: Should contain pickle file
    """
    model.load_state_dict(torch.load(PATH, map_location=device))


# The last two functions are for the plotting.py file and the Plotter class
def get_series_color(y):
    if np.average(y) >= 0.95:
        return "black"
    elif np.average(y) == 0.0:
        return "black"
    else:
        return "black"


def get_y_height(y):
    if np.average(y) >= 0.95:
        return 1.5
    elif np.average(y) == 0.0:
        return 0.1
    else:
        return max(y) + 0.1