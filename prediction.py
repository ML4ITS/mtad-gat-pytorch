from tqdm import tqdm

from utils import *
from eval_methods import pot_eval


class Predictor:
	""" MTAD-GAT predictor class.

	    :param model: MTAD-GAT model (pre-trained) used to forecast and reconstruct
	    :param window_size: Length of the input sequence
	    :param n_features: Number of input features
	    :param gamma: weighting of recon loss relative to prediction loss (1=equally weighted)
	    :param batch_size: Number of windows in a single batch
	    :param boolean use_cuda: To be run on GPU or not

	    """
	def __init__(self, model, window_size, n_features, gamma=1, batch_size=256, use_cuda=True, save_path=''):
		self.model = model
		self.window_size = window_size
		self.n_features = n_features
		self.gamma = gamma
		self.batch_size = batch_size
		self.use_cuda = use_cuda
		self.save_path = save_path

	def get_score(self, values):
		""" Method that calculates anomaly score using given model and data
			:param values: 2D array of multivariate time series data, shape (n, k)
		"""

		print("Predicting and calculating anomaly scores..")
		data = SlidingWindowDataset(values, self.window_size)
		# loader, _, _ = create_data_loaders(data, val_split=0.0, batch_size=, shuffle=False)
		loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
		device = 'cuda' if self.use_cuda and torch.cuda.is_available() else 'cpu'

		self.model.eval()
		preds = []
		recons = []
		with torch.no_grad():
			for x, y in tqdm(loader):
				x = x.to(device)
				y = y.to(device)

				y_hat, _ = self.model(x)

				# Shifting input to include the observed value (y) when doing the reconstruction
				recon_x = torch.cat((x[:, 1:, :], y), dim=1)
				_, window_recon = self.model(recon_x)

				preds.append(y_hat.detach().cpu().numpy())
				recons.append(window_recon[:, -1, :].detach().cpu().numpy())

		preds = np.concatenate(preds, axis=0)
		recons = np.concatenate(recons, axis=0)
		actual = values.detach().cpu().numpy()[self.window_size:]

		anomaly_scores = np.mean(np.sqrt((preds - actual) ** 2) + self.gamma * np.sqrt((recons - actual) ** 2), 1)

		return anomaly_scores

	def predict_anomalies(self, train, test, true_anomalies, save_scores=False, load_scores=False):
		""" Predicts anomalies for given test set.
			Train data needed to setting threshold (via the peak-over-threshold method)

			:param train: 2D array of train multivariate time series data
			:param test: 2D array of test multivariate time series data
		"""

		if load_scores:
			print('Loading anomaly scores')
			train_anomaly_scores = np.load(f'{self.save_path}/train_scores.npy')
			test_anomaly_scores = np.load(f'{self.save_path}/test_scores.npy')
		else:
			train_anomaly_scores = self.get_score(train)
			test_anomaly_scores = self.get_score(test)

			if save_scores:
				np.save(f'{self.save_path}/train_scores', train_anomaly_scores)
				np.save(f'{self.save_path}/test_scores', test_anomaly_scores)
				print(f'Anomaly scores saved to {self.save_path}/<train/test>_scores.npy')

		print("Running POT..")
		eval = pot_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies,
						q=1e-5, level=0.999)
		print_eval = dict(eval)
		del print_eval['pred']
		del print_eval['pot_thresholds']
		print(str(print_eval))

		df = pd.DataFrame()
		df['a_score'] = test_anomaly_scores
		df['pot_threshold'] = eval['pot_thresholds']
		df['pred_anomaly'] = eval['pred'].astype(int)
		df['anomaly'] = true_anomalies

		df_path = f'{self.save_path}/test_pot.pkl'
		print(f'Saving output to {df_path}')
		df.to_pickle(f'{df_path}')
		print('-- Done.')


