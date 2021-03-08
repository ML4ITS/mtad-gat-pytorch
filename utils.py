import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


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
		return len(self.data) - self.window


def minibatch_slices_iterator(length, batch_size,
							  ignore_incomplete_batch=False):
	"""
	Iterate through all the mini-batch slices.

	Args:
		length (int): Total length of data in an epoch.
		batch_size (int): Size of each mini-batch.
		ignore_incomplete_batch (bool): If :obj:`True`, discard the final
			batch if it contains less than `batch_size` number of items.
			(default :obj:`False`)

	Yields
		slice: Slices of each mini-batch.  The last mini-batch may contain
			   less indices than `batch_size`.
	"""
	start = 0
	stop1 = (length // batch_size) * batch_size
	while start < stop1:
		yield slice(start, start + batch_size, 1)
		start += batch_size
	if not ignore_incomplete_batch and start < length:
		yield slice(start, length, 1)


class BatchSlidingWindow(object):
	"""
	Class for obtaining mini-batch iterators of sliding windows.

	Each mini-batch will have `batch_size` windows.  If the final batch
	contains less than `batch_size` windows, it will be discarded if
	`ignore_incomplete_batch` is :obj:`True`.

	Args:
		array_size (int): Size of the arrays to be iterated.
		window_size (int): The size of the windows.
		batch_size (int): Size of each mini-batch.
		excludes (np.ndarray): 1-D `bool` array, indicators of whether
			or not to totally exclude a point.  If a point is excluded,
			any window which contains that point is excluded.
			(default :obj:`None`, no point is totally excluded)
		shuffle (bool): If :obj:`True`, the windows will be iterated in
			shuffled order. (default :obj:`False`)
		ignore_incomplete_batch (bool): If :obj:`True`, discard the final
			batch if it contains less than `batch_size` number of windows.
			(default :obj:`False`)
	"""

	def __init__(self, array_size, window_size, batch_size, excludes=None,
				 shuffle=False, ignore_incomplete_batch=False):
		# check the parameters
		if window_size < 1:
			raise ValueError('`window_size` must be at least 1')
		if array_size < window_size:
			raise ValueError('`array_size` must be at least as large as '
							 '`window_size`')
		if excludes is not None:
			excludes = np.asarray(excludes, dtype=np.bool)
			expected_shape = (array_size,)
			if excludes.shape != expected_shape:
				raise ValueError('The shape of `excludes` is expected to be '
								 '{}, but got {}'.
								 format(expected_shape, excludes.shape))

		# compute which points are not excluded
		if excludes is not None:
			mask = np.logical_not(excludes)
		else:
			mask = np.ones([array_size], dtype=np.bool)
		mask[: window_size - 1] = False
		where_excludes = np.where(excludes)[0]
		for k in range(1, window_size):
			also_excludes = where_excludes + k
			also_excludes = also_excludes[also_excludes < array_size]
			mask[also_excludes] = False

		# generate the indices of window endings
		indices = np.arange(array_size)[mask]
		self._indices = indices.reshape([-1, 1])

		# the offset array to generate the windows
		self._offsets = np.arange(-window_size + 1, 1)

		# memorize arguments
		self._array_size = array_size
		self._window_size = window_size
		self._batch_size = batch_size
		self._shuffle = shuffle
		self._ignore_incomplete_batch = ignore_incomplete_batch

	def get_iterator(self, arrays):
		"""
		Iterate through the sliding windows of each array in `arrays`.

		This method is not re-entrant, i.e., calling :meth:`get_iterator`
		would invalidate any previous obtained iterator.

		Args:
			arrays (Iterable[np.ndarray]): 1-D arrays to be iterated.

		Yields:
			tuple[np.ndarray]: The windows of arrays of each mini-batch.
		"""
		# check the parameters
		arrays = tuple(np.asarray(a) for a in arrays)
		if not arrays:
			raise ValueError('`arrays` must not be empty')

		# shuffle if required
		if self._shuffle:
			np.random.shuffle(self._indices)

		# iterate through the mini-batches
		for s in minibatch_slices_iterator(
				length=len(self._indices),
				batch_size=self._batch_size,
				ignore_incomplete_batch=self._ignore_incomplete_batch):
			idx = self._indices[s] + self._offsets
			yield tuple(a[idx] if len(a.shape) == 1 else a[idx, :] for a in arrays)