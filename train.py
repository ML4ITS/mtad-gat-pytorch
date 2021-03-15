import torch.nn as nn
import argparse
import json

from utils import *
from mtad_gat import MTAD_GAT
from training import Trainer
from prediction import Predictor


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Data params
	parser.add_argument('--dataset', type=str, default='smd')
	parser.add_argument('--group', type=str, default="1-1",
						help='Required for smd dataset. <group_index>-<index>')
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
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--bs', type=int, default=256)
	parser.add_argument('--init_lr', type=float, default=1e-3)
	parser.add_argument('--val_split', type=float, default=0.1)
	parser.add_argument('--shuffle_dataset', type=bool, default=True)
	parser.add_argument('--dropout', type=float, default=0.3)
	parser.add_argument('--use_cuda', type=bool, default=True)
	parser.add_argument('--model_path', type=str, default="models")
	parser.add_argument('--print_every', type=int, default=1)

	# Other
	parser.add_argument('--comment', type=str, default="")

	args = parser.parse_args()

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
	args_summary = str(args.__dict__)

	if args.dataset == 'smd':
		(x_train, _), (x_test, y_test) = get_data(f'machine-{group_index}-{index}')
	else:
		(x_train, _), (x_test, y_test) = get_data(args.dataset)

	x_train = torch.from_numpy(x_train).float()
	x_test = torch.from_numpy(x_test).float()
	n_features = x_train.shape[1]

	train_dataset = SlidingWindowDataset(x_train, window_size)
	test_dataset = SlidingWindowDataset(x_test, window_size)

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
					  model_path, log_dir, print_every, args_summary)

	trainer.fit(train_loader, val_loader)

	plot_losses(trainer.losses, save_path=output_path)

	# Check test loss
	test_loss = trainer.evaluate(test_loader)
	print(f'Test forecast loss: {test_loss[0]:.5f}')
	print(f'Test reconstruction loss: {test_loss[1]:.5f}')
	print(f'Test total loss: {test_loss[2]:.5f}')

	# Predict anomalies
	# 'level' argument for POT-method
	level_dict = {'smap': 0.93, 'msl': 0.99, 'smd-1': 0.9950, 'smd-2': 0.9925, 'smd-3': 0.9999}
	key = 'smd-' + args.group[0] if args.dataset == 'smd' else args.dataset
	level = level_dict[key]

	trainer.load(f'{model_path}/{trainer.id}/{trainer.id}_model.pt')
	best_model = trainer.model
	predictor = Predictor(best_model, window_size, n_features, batch_size=256, level=level, gamma=0.8, save_path=output_path)
	label = y_test[window_size:]
	predictor.predict_anomalies(x_train, x_test, label, save_scores=True)

	# Save config
	args_path = f'{model_path}/{trainer.id}/{trainer.id}_config.txt'
	with open(args_path, 'w') as f:
		json.dump(args.__dict__, f, indent=2)








