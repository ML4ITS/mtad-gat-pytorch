from tqdm import tqdm

from eval_methods import *
from utils import *

import pandas as pd
import json


class Predictor:
    """MTAD-GAT predictor class.

    :param model: MTAD-GAT model (pre-trained) used to forecast and reconstruct
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param level: param used in the Peak-Over-Threshold method
    :param gamma: weighting of recon loss relative to prediction loss (1=equally weighted)
    :param batch_size: Number of windows in a single batch
    :param boolean use_cuda: To be run on GPU or not
    :param save_path: path to save predictions and other output files

    """

    def __init__(self, model, window_size, n_features, pred_args):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = pred_args['target_dims']
        self.q = pred_args['q']
        self.level = pred_args['level']
        self.use_mov_av = pred_args['use_mov_av']
        self.gamma = pred_args['gamma']
        self.save_path = pred_args['save_path']
        self.batch_size = 256
        self.use_cuda = True
        self.pred_args = pred_args

    def get_score(self, values, save_forecasts_and_recons=False, save_name=''):
        """Method that calculates anomaly score using given model and data
        :param values: 2D array of multivariate time series data, shape (N, k)
        :param save_forecasts_and_recons: if True, saves forecasts and
        reconstructions together with anomaly score for each feature
        """

        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

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

        if self.target_dims is not None:
            actual = actual[:, self.target_dims]

        if save_forecasts_and_recons:
            df = pd.DataFrame()
            for i in range(preds.shape[1]):
                df[f"Pred_{i}"] = preds[:, i]
                df[f"Recon_{i}"] = recons[:, i]
                df[f"True_{i}"] = actual[:, i]
                df[f"A_Score_{i}"] = np.sqrt((preds[:, i] - actual[:, i]) ** 2) \
                                    + self.gamma * np.sqrt((recons[:, i] - actual[:, i]) ** 2)

            df_path = f"{self.save_path}/{save_name}.pkl"
            print(f"Saving feature forecasts, reconstructions and anomaly scores to {df_path}")
            df.to_pickle(f"{df_path}")

        anomaly_scores = np.mean(np.sqrt((preds - actual) ** 2) + self.gamma * np.sqrt((recons - actual) ** 2), 1)

        return anomaly_scores

    def predict_anomalies(self, train, test, true_anomalies, save_scores=False, load_scores=False):
        """Predicts anomalies for given test set.
        Train data needed to setting threshold (via the peak-over-threshold method)

        :param train: 2D array of train multivariate time series data
        :param test: 2D array of test multivariate time series data
        """

        if load_scores:
            print("Loading anomaly scores")
            train_anomaly_scores = np.load(f"{self.save_path}/train_scores.npy")
            test_anomaly_scores = np.load(f"{self.save_path}/test_scores.npy")
        else:
            train_anomaly_scores = self.get_score(train, save_forecasts_and_recons=True, save_name='preds_train')
            test_anomaly_scores = self.get_score(test, save_forecasts_and_recons=True, save_name='preds_test')

        if save_scores:
            np.save(f"{self.save_path}/train_scores", train_anomaly_scores)
            np.save(f"{self.save_path}/test_scores", test_anomaly_scores)
            print(f"Anomaly scores saved to {self.save_path}/<train/test>_scores.npy")

        # Recommended values for start, end
        # SMD: 0.01, 0.5
        # MSL: 0.1, 2
        # SMAP:
        if true_anomalies is not None:
            bf_eval = bf_search(test_anomaly_scores, true_anomalies, start=0.01, end=5, step_num=100, verbose=False)
            print(f'Results using best f1 score search:\n {bf_eval}')

        if self.use_mov_av:
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()
            # test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

        # output = pd.read_pickle(f'{self.save_path}/preds.pkl')
        # test_anomaly_scores = test_anomaly_scores / output['True_0'].values
        # train_anomaly_scores = train_anomaly_scores / np.ptp(true_anomalies.astype(int))
        # test_anomaly_scores = test_anomaly_scores / np.ptp(true_anomalies.astype(int))

        eval = pot_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies, q=self.q, level=self.level)

        if true_anomalies is not None:
            print_eval = dict(eval)
            del print_eval["pred"]
            del print_eval["thresholds"]
            print(f'Results using peak-over-threshold method:\n {print_eval}')
        else:
            print(f'No labels given, not doing evaluation of predictions')

        df = pd.DataFrame()
        df["a_score"] = test_anomaly_scores
        df["threshold"] = eval["thresholds"]
        df["pred_anomaly"] = eval["pred"].astype(int)
        df["anomaly"] = true_anomalies

        print(f"Saving output to {self.save_path}/")
        df.to_pickle(f"{self.save_path}/anomaly_preds.pkl")

        if true_anomalies is not None:
            for k, v in print_eval.items():
                print_eval[k] = float(v)
            for k, v in bf_eval.items():
                bf_eval[k] = float(v)
        else:
            print_eval = None
            bf_eval = None

        summary = {'pred_args': self.pred_args,
                   'pot_result': print_eval,
                   'bf_result': bf_eval}

        with open(f"{self.save_path}/summary.txt", "w") as f:
            json.dump(summary, f, indent=2)

        print("-- Done.")
