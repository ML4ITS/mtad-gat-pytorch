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
	with torch.no_grad():
		tot_loss = 0
		for x, y in loader:
			y_hat = model(x.squeeze(0))
			loss = torch.sqrt(criterion(y_hat, y.squeeze(0)))
			tot_loss += loss.item()
	return tot_loss / len(loader)


def predict(model, x, true_y, orig_min, orig_max, plot='all'):
	preds = []
	model.eval()
	with torch.no_grad():
		for i in range(x.shape[0]):
			pred = model(x[i])
			preds.append(pred.detach().cpu().numpy())

		# If using prediction as next value instead of true
		#if i < x.shape[0]-1:
		#	x[i+1, -1, :] = preds

	preds = np.array(preds)
	true_y = true_y.detach().cpu().numpy()

	# Denormalize before plot and mse
	# preds = denormalize(preds, orig_min, orig_max)
	# true_y = denormalize(true_y, orig_min, orig_max)

	# Plot preds and true
	for i in range(preds.shape[2]):
		plt.plot([j for j in range(len(preds))], preds[:, 0, i].ravel(), label='Preds')
		plt.plot([j for j in range(len(true_y))], true_y[:, 0, i].ravel(), label='True')
		plt.title(f'Feature: {i}')
		plt.legend()
		plt.show()

	mse = mean_squared_error(true_y.squeeze(), preds.squeeze())
	return mse


if __name__ == '__main__':

	window_size = 5
	horizon = 1

	data = process_gas_sensor_data(window_size, horizon)
	feature_names = data['feature_names']

	train_x = torch.from_numpy(data['train_x']).float()
	train_y = torch.from_numpy(data['train_y']).float()

	val_x = torch.from_numpy(data['val_x']).float()
	val_y = torch.from_numpy(data['val_y']).float()

	test_x = torch.from_numpy(data['test_x']).float()
	test_y = torch.from_numpy(data['test_y']).float()

	train_x_min = data['train_x_min']
	train_x_max = data['train_x_max']

	print(train_x.shape)
	print(test_x.shape)

	num_nodes = len(feature_names)

	n_epochs = 5
	model = MTAD_GAT(num_nodes, window_size, horizon, dropout=0.0)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	if torch.cuda.is_available():
		model.cuda()

	criterion = nn.MSELoss()

	batch_size = 1
	train_data = TensorDataset(train_x, train_y)
	train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=False)

	val_data = TensorDataset(val_x, val_y)
	val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, drop_last=False)

	test_data = TensorDataset(test_x, test_y)
	test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=False)

	init_train_loss = evaluate(model, train_loader, criterion)
	print(f'Init train loss: {init_train_loss}')

	init_val_loss = evaluate(model, val_loader, criterion)
	print(f'Init val loss: {init_val_loss}')

	train_losses = []
	epoch_losses = []
	for epoch in range(n_epochs):
		model.train()
		model.set_gru_init_hidden(window_size)
		avg_loss = 0
		for x, y in train_loader:
			optimizer.zero_grad()
			y_hat = model(x.squeeze(0))
			loss = torch.sqrt(criterion(y_hat, y.squeeze(0)))
			loss.backward()
			optimizer.step()

			train_losses.append(loss.item())
			avg_loss += loss.item()

		avg_loss /= len(train_loader)
		epoch_losses.append(avg_loss)

		# Evaluate on validation set
		val_loss = evaluate(model, val_loader, criterion)

		print(f'[Epoch {epoch+1}] Train loss: {avg_loss:.5f}, Val loss: {val_loss:.5f}')

	plt.plot([i for i in range(len(epoch_losses))], epoch_losses)
	plt.show()

	# Predict
	mse_train = predict(model, train_x, train_y, train_x_min, train_x_max)
	mse_val = predict(model, val_x, val_y, train_x_min, train_x_max)
	mse_test = predict(model, test_x, test_y, train_x_min, train_x_max)

	test_loss = evaluate(model, test_loader, criterion)
	print(f'Test loss: {test_loss:.3f}')
	print(f'Test mse: {mse_test:.3f}')









