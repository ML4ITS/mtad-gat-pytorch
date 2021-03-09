import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse

from utils import *
from mtad_gat import MTAD_GAT
from training import Trainer


def detect_anomalies(model, loader, save_path, true_anomalies=None, use_cuda=True):
	""" Method that forecasts next value and reconstructs input using given model.
		Saves dataframe that, for each timestamp, contains:
			- predicted value for each feature
			- reconstructed value for each feature 
			- true value for each feature
			- RSE between predicted and true value
			- if timestamp is predicted anomaly (0 or 1)
			- whether the timestamp was an anomaly (if provided)
			
		:param model: Model (pre-trained) used to forecast and reconstruct
		:param loader: Pytorch dataloader
		:param save_path: Path to save output
		:param true_anomalies: boolean array indicating if timestamp is anomaly (0 or 1)
	"""
	print(f'Detecting anomalies..')
	model.eval()

	preds = []
	true_y = []
	recons = []

	device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)

			y_hat, window_recons = model(x)
			if y_hat.ndim == 3:
				y_hat = y_hat.squeeze(1)
			if y.ndim == 3:
				y = y.squeeze(1)

			preds.extend(y_hat.detach().cpu().numpy())
			true_y.extend(y.detach().cpu().numpy())
			recons.extend(window_recons.detach().cpu().numpy())

	window_size = x.shape[1]
	n_features = x.shape[2]

	preds = np.array(preds)
	true_y = np.array(true_y)
	recons = np.array(recons)

	last_recons = recons[-1, -(recons.shape[0] % window_size)+1:, :]

	recons = recons[window_size::window_size].reshape((-1, n_features))
	recons = np.append(recons, last_recons, axis=0)
	recons = np.append(recons, [true_y[-1, :]], axis=0)

	rmse = np.sqrt(mean_squared_error(true_y, preds)) + np.sqrt(mean_squared_error(true_y, recons))
	#l1 = np.abs(recons-true_y).mean()
	print(rmse.mean())
	gamma = 1

	df = pd.DataFrame()
	for i in range(n_features):
		df[f'Pred_{i}'] = preds[:, i]
		df[f'Recon_{i}'] = recons[:, i]
		df[f'True_{i}'] = true_y[:, i]
		df[f'A_Score_{i}'] = np.sqrt((preds[:, i] - true_y[:, i]) ** 2) + gamma * np.sqrt((recons[:, i] - true_y[:, i]) ** 2)
		# df[f'F_Loss_{i}'] = np.sqrt((preds[:, i] - true_y[:, i]) ** 2)
		# df[f'R_Loss_{i}'] = np.sqrt((recons[:, i] - true_y[:, i]) ** 2)
		# df[f'R_Loss_{i}'] = np.abs(recons[:, i] - recons_true[:, i])

	df['Pred_Anomaly'] = -1  # TODO: Implement threshold method for anomaly
	df['True_Anomaly'] = true_anomalies[window_size:] if true_anomalies is not None else 0

	print(f'Saving output to {save_path}')
	df.to_pickle(f'{save_path}.pkl')
	print('-- Done.')

def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, val_dataset=None, test_dataset=None):
	train_loader, val_loader, test_loader = None, None, None
	if val_split == 0.0:
		print(f'train_size: {len(train_dataset)}')
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

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
												   sampler=train_sampler)
		val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
												 sampler=valid_sampler)

		print(f'train_size: {len(train_indices)}')
		print(f'validation_size: {len(val_indices)}')

	if test_dataset is not None:
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
		print(f'test_size: {len(val_indices)}')

	return train_loader, val_loader, test_loader

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Data params
	parser.add_argument('--dataset', type=str, default='smd')
	parser.add_argument('--group', type=str, default="1-1",
						help='Needed for smd dataset. <group_index>-<index>')
	parser.add_argument('--lookback', type=int, default=100)
	parser.add_argument('--horizon', type=int, default=1)
	parser.add_argument('--target_col', type=int, default=None)

	# Model params
	parser.add_argument('--kernel_size', type=int, default=7)
	parser.add_argument('--gru_layers', type=int, default=1)
	parser.add_argument('--gru_hid_dim', type=int, default=150)
	parser.add_argument('--autoenc_layers', type=int, default=1)
	parser.add_argument('--autoenc_hid_dim', type=int, default=128)
	parser.add_argument('--fc_layers', type=int, default=3)
	parser.add_argument('--fc_hid_dim', type=int, default=150)

	# Train params
	parser.add_argument('--test_size', type=float, default=0.2)
	parser.add_argument('--epochs', type=int, default=30)
	parser.add_argument('--bs', type=int, default=256)
	parser.add_argument('--init_lr', type=float, default=1e-3)
	parser.add_argument('--val_split', type=float, default=0.1)
	parser.add_argument('--shuffle_dataset', type=bool, default=True)
	parser.add_argument('--dropout', type=float, default=0.3)
	parser.add_argument('--use_cuda', type=bool, default=True)
	parser.add_argument('--model_path', type=str, default="models/")
	parser.add_argument('--print_every', type=int, default=1)

	args = parser.parse_args()
	print(args)

	if args.dataset == 'smd':
		output_path = f'output/smd/{args.group}'
	else:
		output_path = f'output/{args.dataset}'

	log_dir = f'{output_path}/logs'

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	window_size = args.lookback
	horizon = args.horizon
	target_col = args.target_col
	n_epochs = args.epochs
	batch_size = args.bs
	init_lr = args.init_lr
	val_split = args.val_split
	shuffle_dataset = args.shuffle_dataset
	use_cuda = args.use_cuda
	model_path = args.model_path
	print_every = args.print_every
	group_index = args.group[0]
	index = args.group[2]

	(x_train, _), (x_test, y_test) = get_data(f'machine-{group_index}-{index}')

	x_train = torch.from_numpy(x_train).float()
	x_test = torch.from_numpy(x_test).float()
	n_features = x_train.shape[1]

	train_dataset = SMDDataset(x_train, window_size)
	test_dataset = SMDDataset(x_test, window_size)

	train_loader, val_loader, test_loader = create_data_loaders(train_dataset, batch_size, val_split, shuffle_dataset,
																test_dataset=test_dataset)

	model = MTAD_GAT(n_features, window_size, horizon, n_features, batch_size,
					 kernel_size=args.kernel_size,
					 dropout=args.dropout,
					 gru_n_layers=args.gru_layers,
					 gru_hid_dim=args.gru_hid_dim,
					 autoenc_n_layers=args.autoenc_layers,
					 autoenc_hid_dim=args.autoenc_hid_dim,
					 forecast_n_layers=args.fc_layers,
					 forecast_hid_dim=args.fc_hid_dim,
					 use_cuda=args.use_cuda)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
	forecast_criterion = nn.MSELoss()
	recon_criterion = nn.MSELoss()

	trainer = Trainer(model, optimizer, window_size, n_features, n_epochs, batch_size,
					  init_lr, forecast_criterion, recon_criterion, use_cuda,
					  model_path, log_dir, print_every)

	trainer.fit(train_loader, val_loader)

	plot_losses(trainer.losses, save_path=output_path)

	# Creating non-shuffled train loader
	train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)

	test_loss = trainer.evaluate(test_loader)
	print(f'Test forecast loss: {test_loss[0]:.5f}')
	print(f'Test reconstruction loss: {test_loss[1]:.5f}')
	print(f'Test total loss: {test_loss[2]:.5f}')

	detect_anomalies(model, train_loader, save_path=f'{output_path}/train_out', use_cuda=use_cuda)
	detect_anomalies(model, test_loader, save_path=f'{output_path}/test_out', true_anomalies=y_test, use_cuda=use_cuda)









