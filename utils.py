import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


prefix = "ServerMachineDataset/processed"


def preprocess(df):
	"""returns normalized and standardized data.
	"""

	df = np.asarray(df, dtype=np.float32)

	if len(df.shape) == 1:
		raise ValueError('Data must be a 2-D array')

	if np.any(sum(np.isnan(df)) != 0):
		print('Data contains null values. Will be replaced with 0')
		df = np.nan_to_num()

	# normalize data
	# df = MinMaxScaler().fit_transform(df)
	print('Data normalized')

	return df


def process_data(dataset_name, window_size=50, horizon=1, test_size=0.2, target_col=None):
	"""

	:param window_size: The number of timestamps to use to forecast
	:param horizon: The number of timestamps to forecast following each window
	:param test_size: Number of timestamps used for test
	:param target_col: If having one particular column as target. If -1 then every column is the target
	:return: dict consisting of feature names, x, and y
	"""
	path = 'datasets'
	df = None
	if dataset_name == 'hpc':
		df = pd.read_csv(f'{path}/household_power_consumption_hourly.csv', delimiter=',')
		# df.drop(['Date', 'Hour', 'Datetime'], axis=1, inplace=True)
		df.drop(['Date', 'Hour', 'Datetime', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis=1, inplace=True)
	elif dataset_name == 'gsd':
		df = pd.read_csv(f'{path}/gas_sensor_data.csv', delimiter=',')
		df.drop(['Time', 'Temperature', 'Rel_Humidity'], axis=1, inplace=True)

	#df = pd.read_csv('datasets/household_power_consumption_hourly.csv', delimiter=',')
	#df.drop(['Date', 'Hour', 'Datetime'], axis=1, inplace=True)
	# df.drop(['Date', 'Hour', 'Datetime', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis=1, inplace=True)

	n = df.shape[0]
	print(n)
	values = df.values
	feature_names = df.columns.tolist()

	scaler = MinMaxScaler()
	values = scaler.fit_transform(values)

	# Create forecasting dataset
	x, y = [], []
	for i in range(n - window_size - horizon):
		window_end = i + window_size
		horizon_end = window_end + horizon
		x_i = values[i:window_end, :]

		if target_col is not None:
			y_i = values[window_end:horizon_end, target_col]

		else:
			y_i = values[window_end:horizon_end, :]

		x.append(x_i)
		y.append(y_i)

	# if target_col is not None:

	# Splitting in train, val, test
	test_start = int(n - test_size * n)
	val_start = int(test_start - 0.1 * test_start)  # Validation size: 10% of training data

	# train_end = len(x) - val_size - test_size
	train_x = np.array(x[:val_start])
	train_y = np.array(y[:val_start])

	val_x = np.array(x[val_start:test_start])
	val_y = np.array(y[val_start:test_start])

	test_x = np.array(x[test_start:])
	test_y = np.array(y[test_start:])

	print(f'Total samples: {len(x)}')
	print(f'# of training sampels: {len(train_x)}')
	print(f'# of validation sampels: {len(val_x)}')
	print(f'# of test sampels: {len(test_x)}')

	print("-- Processing done.")

	return {'feature_names': feature_names,
			'train_x': train_x,
			'train_y': train_y,
			'val_x': val_x,
			'val_y': val_y,
			'test_x': test_x,
			'test_y': test_y,
			'scaler': scaler}


def get_data_dim(dataset):
	if dataset == 'SMAP':
		return 25
	elif dataset == 'MSL':
		return 55
	elif str(dataset).startswith('machine'):
		return 38
	else:
		raise ValueError('unknown dataset '+str(dataset))


def get_data(dataset, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0,
			 test_start=0):
	"""
	get data from pkl files

	return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
	"""
	if max_train_size is None:
		train_end = None
	else:
		train_end = train_start + max_train_size
	if max_test_size is None:
		test_end = None
	else:
		test_end = test_start + max_test_size
	print('load data of:', dataset)
	print("train: ", train_start, train_end)
	print("test: ", test_start, test_end)
	x_dim = get_data_dim(dataset)
	f = open(os.path.join(prefix, dataset + '_train.pkl'), "rb")
	train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
	f.close()
	try:
		f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
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
	print("test set label shape: ", test_label.shape)
	return (train_data, None), (test_data, test_label)


class SMDDataset(Dataset):
	def __init__(self, data, window, horizon=1):
		self.data = data
		self.window = window
		self.horizon = horizon

	def __getitem__(self, index):
		x = self.data[index:index+self.window]
		y = self.data[index+self.window:index+self.window+self.horizon]
		return x, y

	def __len__(self):
		return len(self.data) - self.window #- self.horizon


def plot_losses(losses, save_path=''):
	"""
	:param losses: dict with losses
	:param save_path: path where plots get saved
	"""

	plt.plot(losses['train_forecast'], label='Forecast loss')
	plt.plot(losses['train_recon'], label='Recon loss')
	plt.plot(losses['train_total'], label='Total loss')
	plt.title('Training losses during training')
	plt.xlabel("Epoch")
	plt.legend()
	plt.savefig(f'{save_path}/train_losses.png', bbox_inches='tight')
	plt.show()
	plt.close()

	plt.plot(losses['val_forecast'], label='Forecast loss')
	plt.plot(losses['val_recon'], label='Recon loss')
	plt.plot(losses['val_total'], label='Total loss')
	plt.title('Validation losses during training')
	plt.xlabel("Epoch")
	plt.legend()
	plt.savefig(f'{save_path}/validation_losses.png', bbox_inches='tight')
	plt.show()
	plt.close()