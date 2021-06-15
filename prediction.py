from tqdm import tqdm
import pandas as pd
import json

from eval_methods import *
from utils import *


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

    def __init__(self, model, window_size, n_features, pred_args, summary_file_name="summary.txt"):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = pred_args["target_dims"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.batch_size = 256
        self.use_cuda = True
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name

        self.preds_train = None
        self.preds_test = None

    def get_score(self, values):
        """Method that calculates anomaly score using given model and data
        :param values: 2D array of multivariate time series data, shape (N, k)
        :return np array of anomaly scores + dataframe with prediction for each channel and global anomalies
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
                # Extract last reconstruction only
                recons.append(window_recon[:, -1, :].detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        recons = np.concatenate(recons, axis=0)
        actual = values.detach().cpu().numpy()[self.window_size :]

        if self.target_dims is not None:
            actual = actual[:, self.target_dims]

        df = pd.DataFrame()
        for i in range(preds.shape[1]):
            df[f"Forecast_{i}"] = preds[:, i]
            df[f"Recon_{i}"] = recons[:, i]
            df[f"True_{i}"] = actual[:, i]
            df[f"A_Score_{i}"] = np.sqrt((preds[:, i] - actual[:, i]) ** 2) + self.gamma * np.sqrt(
                (recons[:, i] - actual[:, i]) ** 2
            )
        anomaly_scores = np.mean(np.sqrt((preds - actual) ** 2) + self.gamma * np.sqrt((recons - actual) ** 2), 1)

        return anomaly_scores, df

    def predict_anomalies(self, train, test, true_anomalies, save_scores=False, load_scores=False, save_output=True):
        """Predicts anomalies for given test set.
        Train data needed to set threshold

        :param train: 2D array of train multivariate time series data
        :param test: 2D array of test multivariate time series data
        :param true_anomalies: true anomalies of test set, None if not available
        :param save_scores: Whether to save anomaly scores of train and test
        :param load_scores: Whether to load anomaly scores instead of calculating them
        :param save_output: Whether to save output dataframe
        """

        if load_scores:
            print("Loading anomaly scores")
            train_anomaly_scores = np.load(f"{self.save_path}/train_scores.npy")
            test_anomaly_scores = np.load(f"{self.save_path}/test_scores.npy")

            train_pred_df = pd.read_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df = pd.read_pickle(f"{self.save_path}/test_output.pkl")

        else:
            train_anomaly_scores, train_pred_df = self.get_score(train)
            test_anomaly_scores, test_pred_df = self.get_score(test)

        if save_scores:
            np.save(f"{self.save_path}/train_scores", train_anomaly_scores)
            np.save(f"{self.save_path}/test_scores", test_anomaly_scores)
            print(f"Anomaly scores saved to {self.save_path}/<train/test>_scores.npy")

        if self.use_mov_av:
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

        # Find threshold and predict anomalies for each feature
        out_dim = self.n_features if self.target_dims is None else len(self.target_dims)
        all_preds = np.zeros((len(test_pred_df), out_dim))
        for i in range(out_dim):
            train_feature_anom_scores = train_pred_df[f"A_Score_{i}"].values
            test_feature_anom_scores = test_pred_df[f"A_Score_{i}"].values
            epsilon = find_epsilon(train_feature_anom_scores, reg_level=2) # Using a high reg_level as it is per-feature

            train_feature_anom_preds = (train_feature_anom_scores >= epsilon).astype(int)
            test_feature_anom_preds = (test_feature_anom_scores >= epsilon).astype(int)

            train_pred_df[f"A_Pred_{i}"] = train_feature_anom_preds
            test_pred_df[f"A_Pred_{i}"] = test_feature_anom_preds

            train_pred_df[f"Thresh_{i}"] = epsilon
            test_pred_df[f"Thresh_{i}"] = epsilon

            all_preds[:, i] = test_feature_anom_preds

        # Evaluate using different threshold methods: brute-force, epsilon and peak-over-treshold
        e_eval = epsilon_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies, reg_level=self.reg_level)
        p_eval = pot_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies,
                          q=self.q, level=self.level, dynamic=self.dynamic_pot)
        if true_anomalies is not None:
            bf_eval = bf_search(test_anomaly_scores, true_anomalies, start=0.01, end=2, step_num=100, verbose=False)
        else:
            bf_eval = {}

        global_epsilon = e_eval["threshold"]
        train_global_epsilon = global_epsilon

        print(f"Results using epsilon method:\n {e_eval}")
        print(f"Results using peak-over-threshold method:\n {p_eval}")
        print(f"Results using best f1 score search:\n {bf_eval}")

        for k, v in e_eval.items():
            if not type(e_eval[k]) == list:
                e_eval[k] = float(v)
        for k, v in p_eval.items():
            if not type(p_eval[k]) == list:
                p_eval[k] = float(v)
        for k, v in bf_eval.items():
            bf_eval[k] = float(v)

        summary = {"epsilon_result": e_eval, "pot_result": p_eval, "bf_result": bf_eval}

        with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:
            json.dump(summary, f, indent=2)

        test_pred_df["A_True_Global"] = true_anomalies
        train_pred_df["Thresh_Global"] = train_global_epsilon
        test_pred_df["Thresh_Global"] = global_epsilon
        train_pred_df[f"A_Pred_Global"] = (train_anomaly_scores >= train_global_epsilon).astype(int)
        test_preds_global = (test_anomaly_scores >= global_epsilon).astype(int)
        if true_anomalies is not None:
            test_preds_global = adjust_predicts(None, true_anomalies, global_epsilon, pred=test_preds_global)

        test_pred_df[f"A_Pred_Global"] = test_preds_global

        if save_output:
            print(f"Saving output to {self.save_path}/<train/test>_output.pkl")
            train_pred_df.to_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")

        print("-- Done.")
