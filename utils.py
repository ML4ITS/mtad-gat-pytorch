import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


def preprocess(df):
	""" Returns normalized and standardized data.
	"""

	df = np.asarray(df, dtype=np.float32)

	if len(df.shape) == 1:
		raise ValueError('Data must be a 2-D array')

	if np.any(sum(np.isnan(df)) != 0):
		print('Data contains null values. Will be replaced with 0')
		df = np.nan_to_num()

	# normalize data
	df = MinMaxScaler().fit_transform(df)
	print('Data normalized')

	return df


def get_data_dim(dataset):
	"""

	:param dataset: Name of dataset
	:return: Number of dimensions in data
	"""
	if dataset == "SMAP":
		return 25
	elif dataset == "MSL":
		return 55
	elif str(dataset).startswith("machine"):
		return 38
	elif dataset == 'TELENOR':
		return 14
	else:
		raise ValueError("unknown dataset " + str(dataset))


def get_target_dims(dataset):
	"""

	:param dataset: Name of dataset
	:return: index of data dimension that should be modeled (forecasted and reconstructed),
			 returns None if all input dimensions should be modeled
	"""
	if dataset == "SMAP":
		return [0]
	elif dataset == "MSL":
		return [0]
	elif dataset == "SMD":
		return None
	elif dataset == "TELENOR":
		return None
	else:
		raise ValueError("unknown dataset " + str(dataset))


def get_data(
	dataset,
	max_train_size=None,
	max_test_size=None,
	print_log=True,
	do_preprocess=False,
	train_start=0,
	test_start=0,
):
	"""
	Get data from pkl files

	return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
	"""
	prefix = "datasets"
	if str(dataset).startswith("machine"):
		prefix += "/ServerMachineDataset/processed"
	elif dataset == "TELENOR":
		prefix += '/telenor/processed'
	else:
		prefix += "/data/processed"

	if max_train_size is None:
		train_end = None
	else:
		train_end = train_start + max_train_size
	if max_test_size is None:
		test_end = None
	else:
		test_end = test_start + max_test_size
	print("load data of:", dataset)
	print("train: ", train_start, train_end)
	print("test: ", test_start, test_end)
	x_dim = get_data_dim(dataset)
	f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
	train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
	f.close()
	try:
		f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
		test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
		f.close()
	except (KeyError, FileNotFoundError):
		test_data = None
	try:
		f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
		test_label = pickle.load(f).reshape((-1))[test_start:test_end]
		f.close()
	except (KeyError, FileNotFoundError):
		test_label = None

	if do_preprocess:
		train_data = preprocess(train_data)
		test_data = preprocess(test_data)

	print("train set shape: ", train_data.shape)
	print("test set shape: ", test_data.shape)
	print("test set label shape: ", None if test_label is None else test_label.shape)
	return (train_data, None), (test_data, test_label)


class SlidingWindowDataset(Dataset):
	def __init__(self, data, window, target_dim=None, horizon=1):
		self.data = data
		self.window = window
		self.target_dim = target_dim
		self.horizon = horizon

	def __getitem__(self, index):
		x = self.data[index:index + self.window]
		y = self.data[index + self.window: index + self.window + self.horizon]
		return x, y

	def __len__(self):
		return len(self.data) - self.window  # - self.horizon


def create_data_loaders(
	train_dataset,
	batch_size,
	val_split=0.1,
	shuffle=True,
	val_dataset=None,
	test_dataset=None,
):
	train_loader, val_loader, test_loader = None, None, None
	if val_split == 0.0:
		print(f"train_size: {len(train_dataset)}")
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

	else:
		dataset_size = len(train_dataset)
		indices = list(range(dataset_size))
		split = int(np.floor(val_split * dataset_size))
		if shuffle:
			# np.random.seed(random_seed)
			np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]

		train_sampler = SubsetRandomSampler(train_indices)
		valid_sampler = SubsetRandomSampler(val_indices)

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
		val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

		print(f"train_size: {len(train_indices)}")
		print(f"validation_size: {len(val_indices)}")

	if test_dataset is not None:
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
		print(f"test_size: {len(test_dataset)}")

	return train_loader, val_loader, test_loader


def plot_losses(losses, save_path=""):
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
	plt.show()
	plt.close()


def load(model, PATH, device="cpu"):
	"""
	Loads the model's parameters from the path mentioned
	:param PATH: Should contain pickle file
	"""
	model.load_state_dict(torch.load(PATH, map_location=device))
