import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse

from utils import *
from mtad_gat import MTAD_GAT


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


def evaluate(model, loader, forecast_criterion, recon_criterion):
	model.eval()

	forecast_losses = []
	recon_losses = []

	with torch.no_grad():
		for x, y in loader:
			y_hat, recons = model(x)
			if y_hat.ndim == 3:
				y_hat = y_hat.squeeze(1)
			if y.ndim == 3:
				y = y.squeeze(1)

			forecast_loss = torch.sqrt(forecast_criterion(y, y_hat))
			recon_loss = recon_criterion(x, recons)

			forecast_losses.append(forecast_loss.item())
			recon_losses.append(recon_loss.item())

	forecast_losses = np.array(forecast_losses)
	recon_losses = np.array(recon_losses)

	forecast_loss = np.sqrt((forecast_losses ** 2).mean())
	recon_loss = recon_losses.mean()
	total_loss = forecast_loss + recon_loss

	return forecast_loss, recon_loss, total_loss


def detect_anomalies(model, loader, save_path, true_anomalies=None):
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
	recons_true = []
	with torch.no_grad():
		for x, y in loader:
			y_hat, window_recons = model(x)
			if y_hat.ndim == 3:
				y_hat = y_hat.squeeze(1)
			if y.ndim == 3:
				y = y.squeeze(1)

			preds.extend(y_hat.detach().cpu().numpy())
			true_y.extend(y.detach().cpu().numpy())
			recons.extend(window_recons.detach().cpu().numpy())
			recons_true.extend(x.detach().cpu().numpy())

	window_size = x.shape[1]
	n_features = x.shape[2]

	preds = np.array(preds)
	true_y = np.array(true_y)
	recons = np.array(recons)
	recons_true = np.array(recons_true)

	last_recons = recons[-1, -(recons.shape[0] % window_size)+1:, :]
	last_true_recons = recons_true[-1, -(recons.shape[0] % window_size)+1:, :]
	last_recons = np.append(last_recons, true_y[-1])

	recons = recons[window_size::window_size].reshape((-1, n_features))
	recons = np.append(recons, last_recons, axis=0)

	recons_true = recons_true[window_size::window_size].reshape((-1, n_features))
	recons_true = np.append(recons_true, last_true_recons, axis=0)

	rmse = np.sqrt(mean_squared_error(true_y, preds))
	l1 = np.abs(recons-recons_true).mean()
	print(rmse+l1)

	print(preds.shape)
	print(true_y.shape)
	print(recons.shape)
	print(recons_true.shape)

	df = pd.DataFrame()
	for i in range(n_features):
		df[f'Pred_{i}'] = preds[:, i]
		df[f'True_{i}'] = true_y[:, i]
		df[f'Recon_{i}'] = recons[:, i]
		df[f'True_Recon_{i}'] = recons_true[:, i]
		df[f'Loss_{i}'] = np.sqrt((preds[:, i] - true_y[:, i]) ** 2) + np.abs(recons[:, i] - recons_true[:, i])

	df['Pred_Anomaly'] = -1  # TODO: Implement threshold method for anomaly
	df['True_Anomaly'] = true_anomalies[window_size:] if true_anomalies is not None else 0

	print(f'Saving output to {save_path}')
	df.to_pickle(f'{save_path}.pkl')
	# df.to_csv(f'{save_path}.csv', index=False)
	print('-- Done.')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Data params
	parser.add_argument('--dataset', type=str, default='smd', choices=['hpc', 'gsd'],
						help='hpc: hourly household power consumption data /n gsd: gas sensor data')
	parser.add_argument('--group', type=str, default="1-1",
						help='<group_index>-<index>')
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
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--dropout', type=float, default=0.3)
	parser.add_argument('--use_cuda', type=bool, default=True)
	parser.add_argument('--model_path', type=str, default="./saved_models/")

	args = parser.parse_args()
	print(args)

	if not os.path.exists(f'plots/{args.dataset}'):
		os.makedirs(f'plots/{args.dataset}')

	if not os.path.exists(f'output/{args.dataset}'):
		os.makedirs(f'output/{args.dataset}')

	window_size = args.lookback
	horizon = args.horizon
	target_col = args.target_col
	n_epochs = args.epochs
	batch_size = args.bs
	group_index = args.group[0]
	index = args.group[2]

	(x_train, _), (x_test, y_test) = get_data(f'machine-{group_index}-{index}')

	cuda = torch.cuda.is_available() and args.use_cuda
	device = 'cuda' if cuda else 'cpu'

	x_train = torch.from_numpy(x_train).float().to(device)
	x_train = x_train[:-3000]
	x_val = x_train[-3000:]
	x_test = torch.from_numpy(x_test).float().to(device)

	print(f'x_train shape: {x_train.shape}')
	print(f'x_val shape: {x_val.shape}')
	print(f'x_test shape: {x_test.shape}')

	x_dim = x_train.shape[1]

	train_dataset = SMDDataset(x_train, window=window_size)
	train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=False)

	val_dataset = SMDDataset(x_val, window=window_size)
	val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, drop_last=False)

	test_dataset = SMDDataset(x_test, window=window_size)
	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=False)

	model = MTAD_GAT(x_dim, window_size, horizon, x_dim, batch_size,
					 kernel_size=args.kernel_size,
					 dropout=args.dropout,
					 gru_n_layers=args.gru_layers,
					 gru_hid_dim=args.gru_hid_dim,
					 autoencoder_n_layers=args.autoenc_layers,
					 autoencoder_hid_dim=args.autoenc_hid_dim,
					 forecasting_n_layers=args.fc_layers,
					 forecasting_hid_dim=args.fc_hid_dim,
					 device=device)

	print(f'Device: {device}')
	if cuda:
		model.cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	forecast_criterion = nn.MSELoss()
	recon_criterion = nn.L1Loss()

	init_train_loss = evaluate(model, train_loader, forecast_criterion, recon_criterion)
	print(f'Init total train loss: {init_train_loss[2]}')

	init_val_loss = evaluate(model, val_loader, forecast_criterion, recon_criterion)
	print(f'Init total val loss: {init_val_loss[2]}')

	losses = {
		'train_total': [],
		'train_forecast': [],
		'train_recon': [],
		'val_total': [],
		'val_forecast': [],
		'val_recon': [],
	}

	print(f'Training model for {n_epochs} epochs..')
	for epoch in range(n_epochs):
		model.train()
		forecast_b_losses = []
		recon_b_losses = []
		for x, y in train_loader:
			optimizer.zero_grad()
			preds, recons = model(x)
			if preds.ndim == 3:
				preds = preds.squeeze(1)
			if y.ndim == 3:
				y = y.squeeze(1)

			forecast_loss = torch.sqrt(forecast_criterion(y, preds))
			recon_loss = recon_criterion(x, recons)
			loss = forecast_loss + recon_loss

			loss.backward()
			optimizer.step()

			forecast_b_losses.append(forecast_loss.item())
			recon_b_losses.append(recon_loss.item())

		forecast_b_losses = np.array(forecast_b_losses)
		recon_b_losses = np.array(recon_b_losses)

		forecast_epoch_loss = np.sqrt((forecast_b_losses**2).mean())
		recon_epoch_loss = recon_b_losses.mean()
		total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

		losses['train_forecast'].append(forecast_epoch_loss)
		losses['train_recon'].append(recon_epoch_loss)
		losses['train_total'].append(total_epoch_loss)

		# Evaluate on validation set
		forecast_val_loss, recon_val_loss, total_val_loss = evaluate(model, val_loader, forecast_criterion, recon_criterion)
		losses['val_forecast'].append(forecast_val_loss)
		losses['val_recon'].append(recon_val_loss)
		losses['val_total'].append(total_val_loss)

		print(f'[Epoch {epoch + 1}] '
			  f'Forecast loss: {forecast_epoch_loss:.5f}, '
			  f'Recon loss: {recon_epoch_loss:.5f}, '
			  f'Total loss: {total_epoch_loss:.5f} ...... '
			  
			  f'Val Forecast loss: {forecast_val_loss:.5f}, '
			  f'Val Recon loss: {recon_val_loss:.5f}, '
			  f'Val Total loss: {total_val_loss:.5f}')

	# Plot Losses
	plot_losses(losses, save_path=f'output/{args.dataset}')

	# Evaluate and Predict
	train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, drop_last=False)
	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=False)

	test_loss = evaluate(model, test_loader, forecast_criterion, recon_criterion)
	print(f'Test forecast loss: {test_loss[0]:.5f}')
	print(f'Test reconstruction loss: {test_loss[1]:.5f}')
	print(f'Test total loss: {test_loss[2]:.5f}')

	detect_anomalies(model, train_loader, save_path=f'output/{args.dataset}/machine-{args.group}_train', )
	detect_anomalies(model, test_loader, save_path=f'output/{args.dataset}/machine-{args.group}_test', true_anomalies=y_test)









