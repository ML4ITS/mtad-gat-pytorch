import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse

from utils import process_gas_sensor_data, denormalize
from mtad_gat import MTAD_GAT


def evaluate(model, loader, criterion):
	model.eval()

	losses = []
	with torch.no_grad():
		for x, y in loader:
			y_hat = model(x)
			loss = criterion(y_hat, y.squeeze(1))
			losses.append(loss.item())

	return np.sqrt(np.array(losses).mean())


def predict(model, x, true_y, scaler, plot_name=''):
	model.eval()

	with torch.no_grad():
		preds = model(x).detach().cpu().numpy().squeeze()

	true_y = true_y.detach().cpu().squeeze().numpy()
	rmse = np.sqrt(mean_squared_error(true_y, preds))
	preds = scaler.inverse_transform(preds)
	true_y = scaler.inverse_transform(true_y)



	# Plot preds and true
	for i in range(preds.shape[1]):
		plt.plot([j for j in range(len(preds))], preds[:, i].ravel(), label='Preds')
		plt.plot([j for j in range(len(true_y))], true_y[:, i].ravel(), label='True')
		plt.title(f'Feature: {i}')
		plt.legend()
		plt.savefig(f'plots/{plot_name}_feature{i}.png', bbox_inches='tight')
		plt.show()

	return rmse


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Model params
	parser.add_argument('--lookback', type=int, default=100)
	parser.add_argument('--horizon', type=int, default=1)
	parser.add_argument('--target_col', type=int, default=None)
	parser.add_argument('--kernel_size', type=int, default=7)
	parser.add_argument('--gru_layers', type=int, default=1)
	parser.add_argument('--gru_hid_dim', type=int, default=64)
	parser.add_argument('--fc_layers', type=int, default=1)
	parser.add_argument('--fc_hid_dim', type=int, default=32)

	# Train params
	parser.add_argument('--test_size', type=float, default=0.2)
	parser.add_argument('--epochs', type=int, default=30)
	parser.add_argument('--bs', type=int, default=64)
	parser.add_argument('--lr', type=int, default=1e-4)
	parser.add_argument('--dropout', type=float, default=0.0)
	parser.add_argument('--use_cuda', type=bool, default=True)
	parser.add_argument('--model_path', type=str, default="./saved_models/")

	args = parser.parse_args()
	print(args)

	if not os.path.exists('plots'):
		os.makedirs('plots')

	window_size = args.lookback
	horizon = args.horizon
	target_col = args.target_col

	data = process_gas_sensor_data(window_size, horizon, test_size=args.test_size, target_col=target_col)
	feature_names = data['feature_names']
	print(feature_names)
	out_dim = len(feature_names) if target_col is None else 1
	cuda = torch.cuda.is_available() and args.use_cuda
	device = 'cuda' if cuda else 'cpu'

	scaler = data['scaler']
	train_x = torch.from_numpy(data['train_x']).float().to(device)
	train_y = torch.from_numpy(data['train_y']).float().to(device)

	val_x = torch.from_numpy(data['val_x']).float().to(device)
	val_y = torch.from_numpy(data['val_y']).float().to(device)

	test_x = torch.from_numpy(data['test_x']).float().to(device)
	test_y = torch.from_numpy(data['test_y']).float().to(device)

	print(f'train_x shape: {train_x.shape}')
	print(f'val_x shape: {val_x.shape}')
	print(f'test_x shape: {test_x.shape}')

	num_nodes = len(feature_names)

	model = MTAD_GAT(num_nodes, window_size, horizon, out_dim,
					 kernel_size=args.kernel_size,
					 dropout=args.dropout,
					 gru_n_layers=args.gru_layers,
					 gru_hid_dim=args.gru_hid_dim,
					 forecasting_n_layers=args.fc_layers,
					 forecasting_hid_dim=args.fc_hid_dim,
					 device=device)
	if cuda:
		model.cuda()
		print(f'Device: {device}')

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	criterion = nn.MSELoss()
	n_epochs = args.epochs
	batch_size = args.bs
	train_data = TensorDataset(train_x, train_y)
	train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

	val_data = TensorDataset(val_x, val_y)
	val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, drop_last=True)

	test_data = TensorDataset(test_x, test_y)
	test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)

	init_train_loss = evaluate(model, train_loader, criterion)
	print(f'Init train loss: {init_train_loss}')

	init_val_loss = evaluate(model, val_loader, criterion)
	print(f'Init val loss: {init_val_loss}')

	train_losses = []
	val_losses = []
	print(f'Training model for {n_epochs} epochs..')
	for epoch in range(n_epochs):
		model.train()
		batch_losses = []
		for x, y in train_loader:
			optimizer.zero_grad()
			y_hat = model(x)
			loss = torch.sqrt(criterion(y_hat, y.squeeze(1)))
			loss.backward()
			optimizer.step()

			batch_losses.append(loss.item())

		epoch_loss = np.array(batch_losses).mean()
		train_losses.append(epoch_loss)

		# Evaluate on validation set
		val_loss = evaluate(model, val_loader, criterion)
		val_losses.append(val_loss)

		print(f'[Epoch {epoch+1}] Train loss: {epoch_loss:.5f}, Val loss: {val_loss:.5f}')

	plt.plot(train_losses, label='training loss')
	plt.plot(val_losses, label='validation loss')
	plt.xlabel("Epoch")
	plt.ylabel("MSE")
	plt.legend()
	plt.savefig(f'plots/losses.png', bbox_inches='tight')
	plt.show()

	# Predict
	rmse_train = predict(model, train_x, train_y, scaler, plot_name='train_preds')
	rmse_val = predict(model, val_x, val_y, scaler, plot_name='val_preds')
	rmse_test = predict(model, test_x, test_y, scaler, plot_name='test_preds')

	print(rmse_test)

	test_loss = evaluate(model, test_loader, criterion)
	print(f'Test loss (RMSE): {test_loss:.3f}')









