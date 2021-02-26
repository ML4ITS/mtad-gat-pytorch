import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def normalize(train, val=None, test=None):
	train_min = train.reshape(-1, train.shape[-1]).min(0)
	train_max = train.reshape(-1, train.shape[-1]).max(0)

	train = (train - train_min) / (train_max - train_min)
	if val is not None:
		val = (val - train_min) / (train_max - train_min)
	if test is not None:
		test = (test - train_min) / (train_max - train_min)

	return train, val, test


def denormalize(normalized_d, min_val, max_val):
	return normalized_d * (max_val - min_val) + min_val


def process_gas_sensor_data(window_size=50, horizon=1, test_size=0.2, target_col=None):
	"""

	:param window_size: The number of timestamps to use to forecast
	:param horizon: The number of timestamps to forecast following each window
	:param test_size: Number of timestamps used for test
	:param target_col: If having one particular column as target. If -1 then every column is the target
	:return: dict consisting of feature names, x, and y
	"""
	# df = pd.read_csv('datasets/gas_sensor_data.csv', delimiter=',')
	# df.drop(['Time', 'Temperature', 'Rel_Humidity'], axis=1, inplace=True)
	df = pd.read_csv('datasets/household_power_consumption_hourly.csv', delimiter=',')
	df.drop(['Date', 'Hour', 'Datetime', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis=1, inplace=True)

	n = df.shape[0]
	print(n)
	values = df.values
	feature_names = df.columns.tolist()

	print(values.min(0), values.max(0))
	scaler = MinMaxScaler()
	values = scaler.fit_transform(values)
	print(values.min(0), values.max(0))

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


# process_gas_sensor_data(window_size=168, horizon=1)
# data = process_gas_sensor_data(window_size=250, horizon=1)
# feature_names = data['feature_names']
# train_x = data['train_x']
# train_y = data['train_y']
#
# val_x = data['val_x']
# val_y = data['val_y']
#
# test_x = data['test_x']
# test_y = data['test_y']
#
# print(train_x.shape)
# print(val_x.shape)
# print(test_x.shape)
#
# s1 = train_x[0, :, 3]
# s1_next = train_y[0, :, 3]
#
# s1_all = np.concatenate((s1, s1_next), axis=0)
#
# plt.plot([i for i in range(len(s1_all))], s1_all)
# plt.title('concated y_hat')
# plt.show()
#
# s2 = train_y[1, :1, 3]
# s1_s2 = np.concatenate((s1, s2), axis=0)
#
# plt.plot([i for i in range(len(s1_s2))], s1_s2)
# plt.show()

