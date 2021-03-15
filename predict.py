import argparse
import json

from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Model and data args
	parser.add_argument('--model', type=str, required=True, help="Name of model to use")
	parser.add_argument('--dataset', type=str, default='smd')
	parser.add_argument('--group', type=str, default="1-1",
						help='Required for smd dataset. <group_index>-<index>')
	parser.add_argument('--use_cuda', type=bool, default=True)
	parser.add_argument('--model_path', type=str, default="models")

	# Predictor args
	parser.add_argument('--save_scores', type=bool, default=True, help="To save anomaly scores predicted.")
	parser.add_argument('--load_scores', type=bool, default=False, help="To use already computed anomaly scores")

	args = parser.parse_args()
	print(args)
	model = args.model

	# Peak-Over-Threshold args
	# Recommend values for `level`:
	# SMAP: 0.93
	# MSL: 0.99
	# SMD group 1: 0.9950
	# SMD group 2: 0.9925
	# SMD group 3: 0.9999
	level_dict = {'smap': 0.93, 'msl': 0.99, 'smd-1': 0.9950, 'smd-2': 0.9925, 'smd-3': 0.9999}
	key = 'smd-' + args.group[0] if args.dataset == 'smd' else args.dataset
	level = level_dict[key]

	pre_trained_model_path = f'models/{model}/{model}'
	# Check that model exist
	if not os.path.isfile(f'{pre_trained_model_path}_model.pt'):
		raise Exception(f'Model <{pre_trained_model_path}_model.pt> does not exist.')

	# Get configs of model
	parser = argparse.ArgumentParser()
	model_args, unknown = parser.parse_known_args()
	print(model_args)
	model_args_path = f'{pre_trained_model_path}_config.txt'
	with open(model_args_path, 'r') as f:
		model_args.__dict__ = json.load(f)
	window_size = model_args.lookback

	# Check that model is trained on specified dataset
	if args.dataset != model_args.dataset:
		raise Exception(f'Model trained on {model_args.dataset}, but asked to predict {args.dataset}.')

	if args.dataset == 'smd' and args.group != model_args.group:
		raise Warning(f'Model trained on smd group {model_args.group}, but asked to predict smd group {args.group}.')

	if args.dataset == 'smd':
		output_path = f'output/smd/{args.group}'
	else:
		output_path = f'output/{args.dataset}'

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	if args.dataset == 'smd':
		group_index = args.group[0]
		index = args.group[2]
		(x_train, _), (x_test, y_test) = get_data(f'machine-{group_index}-{index}')
	else:
		(x_train, _), (x_test, y_test) = get_data(args.dataset)

	save_scores = args.save_scores
	load_scores = args.load_scores

	label = y_test[window_size:]
	x_train = torch.from_numpy(x_train).float()
	x_test = torch.from_numpy(x_test).float()
	n_features = x_train.shape[1]

	train_dataset = SlidingWindowDataset(x_train, window_size)
	test_dataset = SlidingWindowDataset(x_test, window_size)

	model = MTAD_GAT(n_features, window_size, model_args.horizon, n_features, model_args.bs,
					 kernel_size=model_args.kernel_size,
					 dropout=model_args.dropout,
					 gru_n_layers=model_args.gru_layers,
					 gru_hid_dim=model_args.gru_hid_dim,
					 autoenc_n_layers=model_args.autoenc_layers,
					 autoenc_hid_dim=model_args.autoenc_hid_dim,
					 forecast_n_layers=model_args.fc_layers,
					 forecast_hid_dim=model_args.fc_hid_dim,
					 use_cuda=model_args.use_cuda)

	device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'
	model = load(model, f'{pre_trained_model_path}_model.pt', device=device)

	predictor = Predictor(model, window_size, n_features, level=level, save_path=output_path)
	predictor.predict_anomalies(x_train, x_test, label, save_scores=save_scores, load_scores=load_scores)