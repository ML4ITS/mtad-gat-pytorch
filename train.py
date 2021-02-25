import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from utils import process_gas_sensor_data, denormalize
from mtad_gat import MTAD_GAT


def evaluate(model, loader, criterion):
	model.eval()
	# model.set_gru_init_hidden(loader.batch_size)
	with torch.no_grad():
		tot_loss = 0
		for x, y in loader:
			y_hat = model(x)
			loss = torch.sqrt(criterion(y_hat, y.squeeze(1)))
			tot_loss += loss.item()
	return tot_loss / len(loader)


def predict(model, x, true_y, scaler):
	preds = []
	model.eval()
	# model.set_gru_init_hidden(1)
	with torch.no_grad():
		for i in range(x.shape[0]):
			pred = model(x[i].unsqueeze(0))
			preds.append(pred.detach().cpu().numpy())

		# If using prediction as next value instead of true
		#if i < x.shape[0]-1:
		#	x[i+1, -1, :] = preds

	preds = np.array(preds).squeeze()
	true_y = true_y.detach().cpu().squeeze().numpy()

	mse = mean_squared_error(true_y, preds)

	preds = scaler.inverse_transform(preds)
	true_y = scaler.inverse_transform(true_y)

	# Plot preds and true
	for i in range(preds.shape[1]):
		plt.plot([j for j in range(len(preds))], preds[:, i].ravel(), label='Preds')
		plt.plot([j for j in range(len(true_y))], true_y[:, i].ravel(), label='True')
		plt.title(f'Feature: {i}')
		plt.legend()
		plt.show()

	return mse


if __name__ == '__main__':

	window_size = 100
	horizon = 1
	target_col = -1  # -1 for forecasting all inputs

	data = process_gas_sensor_data(window_size, horizon, test_size=0.2, target_col=target_col)
	feature_names = data['feature_names']
	print(feature_names)
	out_dim = len(feature_names) if target_col == -1 else 1

	scaler = data['scaler']

	train_x = torch.from_numpy(data['train_x']).float()
	train_y = torch.from_numpy(data['train_y']).float()

	val_x = torch.from_numpy(data['val_x']).float()
	val_y = torch.from_numpy(data['val_y']).float()

	test_x = torch.from_numpy(data['test_x']).float()
	test_y = torch.from_numpy(data['test_y']).float()

	print(f'train_x shape: {train_x.shape}')
	print(f'val_x shape: {val_x.shape}')
	print(f'test_x shape: {test_x.shape}')

	num_nodes = len(feature_names)

	n_epochs = 30
	model = MTAD_GAT(num_nodes, window_size, horizon, out_dim, dropout=0.1, forecasting_n_layers=1, gru_n_layers=1)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

	if torch.cuda.is_available():
		model.cuda()

	criterion = nn.MSELoss()

	batch_size = 128
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
	epoch_losses = []
	print(f'Training model for {n_epochs} epochs..')
	for epoch in range(n_epochs):
		model.train()
		# model.set_gru_init_hidden(window_size)
		avg_loss = 0
		for x, y in train_loader:
			# model.set_gru_init_hidden(batch_size)
			optimizer.zero_grad()
			y_hat = model(x)
			loss = torch.sqrt(criterion(y_hat, y.squeeze(1)))
			loss.backward()
			optimizer.step()

			train_losses.append(loss.item())
			avg_loss += loss.item()

		avg_loss /= len(train_loader)
		epoch_losses.append(avg_loss)

		# Evaluate on validation set
		val_loss = evaluate(model, val_loader, criterion)
		val_losses.append(val_loss)

		print(f'[Epoch {epoch+1}] Train loss: {avg_loss:.5f}, Val loss: {val_loss:.5f}')

	plt.plot(epoch_losses, label='training loss')
	plt.plot(val_losses, label='validation loss')
	plt.xlabel("Epoch")
	plt.ylabel("RMSE")
	plt.legend()
	plt.show()

	# Predict
	mse_train = predict(model, train_x, train_y, scaler)
	mse_val = predict(model, val_x, val_y, scaler)
	mse_test = predict(model, test_x, test_y, scaler)

	test_loss = evaluate(model, test_loader, criterion)
	print(f'Test loss: {test_loss:.3f}')
	print(f'Test mse: {mse_test:.3f}')









