import pandas as pd
import numpy as np
import os
import json
import plotly as py
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import cufflinks as cf

from eval_methods import find_epsilon
cf.go_offline()


class Plotter:

    """
    Class for visualizing results of anomaly detection.
    Includes visualization of forecasts, reconstructions, anomaly scores, predicted and actual anomalies
    """

    def __init__(self, result_path):
        self.result_path = result_path
        self.data_name = self.result_path.split("/")[-1]
        self.train_data = None
        self.test_data = None
        self._load_results()
        self.labels_available = False
        self.pred_cols = None
        self.feature_error_thresholds = []

        if "TELENOR" in result_path:
            path ='../datasets/telenor/site_data/metadata.txt'
            with open(path) as f:
                input_cols = np.array(json.load(f)["columns"][2:])

            result_summary_path = f'{result_path}/summary.txt'
            with open(result_summary_path) as f:
                target_dims = json.load(f)['pred_args']['target_dims']
                if target_dims is None:
                    self.pred_cols = input_cols
                else:
                    self.pred_cols = input_cols[target_dims]

        # Make column-wise anomaly predictions (based on column-wise thresholding)
        for i in range(len(self.pred_cols)):
            feature_anomaly_scores = self.train_data[f'A_Score_{i}'].values
            epsilon = find_epsilon(feature_anomaly_scores)
            self.feature_error_thresholds.append(epsilon)

    def _load_results(self):
        print(f"Loading results of {self.result_path}")
        anomaly_preds = pd.read_pickle(f"{self.result_path}/anomaly_preds.pkl")
        try:
            anomaly_preds["anomaly"] = anomaly_preds["anomaly"].astype(int)
            self.labels_available = True
        except:
            anomaly_preds["anomaly"] = 0

        try:
            train_output = pd.read_pickle(f"{self.result_path}/preds_train.pkl")
            train_anomaly_scores = np.load(f"{self.result_path}/train_scores.npy")
            train_output["Tot_A_Score"] = train_anomaly_scores
            train_output["Pred_Anomaly"] = 0
            train_output["True_Anomaly"] = 0
            train_output["threshold"] = 0
        except:
            train_output = None
            train_anomaly_scores = None

        test_output = pd.read_pickle(f"{self.result_path}/preds_test.pkl")
        test_anomaly_scores = np.load(f"{self.result_path}/test_scores.npy")
        test_output["Tot_A_Score"] = test_anomaly_scores
        test_output["Pred_Anomaly"] = anomaly_preds["pred_anomaly"]
        test_output["True_Anomaly"] = anomaly_preds["anomaly"]
        test_output["threshold"] = anomaly_preds["threshold"]

        self.train_data = train_output
        self.test_data = test_output

    def result_summary(self):
        path = f"{self.result_path}/summary.txt"
        if not os.path.exists(path):
            print(f"Folder {self.result_path} do not have a summary.txt file")
        try:
            print("Result summary:")
            with open(path) as f:
                result_dict = json.load(f)["pot_result"]
                print(f"\tTP: {result_dict['TP']}")
                print(f"\tTN: {result_dict['TN']}")
                print(f"\tFP: {result_dict['FP']}")
                print(f"\tFN: {result_dict['FN']}")
                print(f"\tAvg latency: {result_dict['latency']:.2f}")
                print()
                print(f"\tPrecison:  {result_dict['precision']:.4f}")
                print(f"\tRecall:    {result_dict['precision']:.4f}")
                print(f"\tF1:        {result_dict['f1']:.4f}")
        except FileNotFoundError as e:
            print(e)
        except Exception:
            print('\tNo results because labels are not available')

    def create_shapes(self, ranges, sequence_type, _min, _max, plot_values, xref=None, yref=None):
        """
        Create shapes for regions to highlight in plotly vizzes (true and
        predicted anomaly sequences). Will plot labeled anomalous ranges if
        available.

        Args:
                ranges (list of tuples): tuple of start and end indices for anomaly
                        sequences for a channel
                sequence_type (str): "predict" if predicted values else
                        "true" if actual values. Determines colors.
                _min (float): min y value of series
                _max (float): max y value of series
                plot_values (dict): dictionary of different series to be plotted
                        (predicted, actual, errors, training data)

        Returns:
                (dict) shape specifications for plotly
        """

        if _max is None:
            _max = max(plot_values["errors"])

        if sequence_type is None:
            color = 'red'
        else:
            color = "red" if sequence_type == "true" else "blue"
        shapes = []

        for r in ranges:
            shape = {
                "type": "rect",
                "x0": r[0] - 5,  # self.config.l_s,
                "y0": _min,
                "x1": r[1] + 5,  # self.config.l_s,
                "y1": _max,
                "fillcolor": color,
                "opacity": 0.08,
                "line": {
                    "width": 0,
                },
            }
            if xref is not None:
                shape['xref'] = xref
                shape['yref'] = yref

            shapes.append(shape)

        return shapes

    def get_anomaly_sequences(self, values):
        splits = np.where(values[1:] != values[:-1])[0] + 1

        a_seqs = []
        for i in range(0, len(splits) - 1, 2):
            a_seqs.append([splits[i], splits[i + 1] - 1])

        if len(splits) % 2 == 1:
            a_seqs.append([splits[-1], len(values) - 1])

        return a_seqs

    def plot_channel(self, channel, plot_train=False, type="test", show_tot_err=False, start=None, end=None, plot_errors=True):
        """Plot forecasting, reconstruction, true value of a specific channel (feature),
        along with the anomaly score for that channel
        """

        test_copy = self.test_data.copy()
        plot_data = [test_copy]

        if plot_train:
            train_copy = self.train_data.copy()
            plot_data.append(train_copy)

        for nr, data_copy in enumerate(plot_data):
            if channel < 0 or f"Pred_{channel}" not in data_copy.columns:
                raise Exception(f"Channel {channel} not present in data.")

            if start is not None and end is not None:
                assert start < end
            if start is not None:
                data_copy = data_copy.iloc[start:, :]
            if end is not None:
                start = 0 if start is None else start
                data_copy = data_copy.iloc[: end - start, :]

            i = channel
            plot_values = {
                "y_forecast": data_copy[f"Pred_{i}"].values,
                "y_recon": data_copy[f"Recon_{i}"].values,
                "y_true": data_copy[f"True_{i}"].values,
                "errors": data_copy[f"Tot_A_Score"].values if show_tot_err else data_copy[f"A_Score_{i}"].values,
                "threshold": data_copy["threshold"].values,
            }

            channel_error_threshold = self.feature_error_thresholds[i]
            channel_anomaly_scores = data_copy[f"A_Score_{i}"].values
            anomaly_preds = (channel_anomaly_scores > channel_error_threshold).astype(int)

            anomaly_sequences = {
                'pred': self.get_anomaly_sequences(anomaly_preds),
                # "pred": self.get_anomaly_sequences(data_copy["Pred_Anomaly"].values),
                "true": self.get_anomaly_sequences(data_copy["True_Anomaly"].values),
            }

            y_min = plot_values["y_true"].min()
            y_max = plot_values["y_true"].max()
            e_max = 2 #plot_values["errors"].max()

            y_min -= 0.1 * y_max
            y_max += 0.5 * y_max
            # e_max += 0.5 * e_max

            # y_shapes = create_shapes(segments, 'true', y_min, y_max, plot_values)
            if self.labels_available:
                y_shapes = self.create_shapes(anomaly_sequences["true"], "true", y_min, y_max, plot_values)
                e_shapes = self.create_shapes(anomaly_sequences["true"], "true", 0, e_max, plot_values)

                y_shapes += self.create_shapes(anomaly_sequences["pred"], "predicted", y_min, y_max, plot_values)
                e_shapes += self.create_shapes(anomaly_sequences["pred"], "predicted", 0, e_max, plot_values)
            else:
                y_shapes = self.create_shapes(anomaly_sequences["pred"], None, y_min, y_max, plot_values)
                e_shapes = self.create_shapes(anomaly_sequences["pred"], None, 0, e_max, plot_values)

            y_df = pd.DataFrame(
                {
                    "y_forecast": plot_values["y_forecast"].reshape(-1, ),
                    "y_recon": plot_values["y_recon"].reshape(-1,),
                    "y_true": plot_values["y_true"].reshape(-1,),
                }
            )

            e_df = pd.DataFrame(
                {
                    "e_s": plot_values["errors"].reshape(-1,),
                    "threshold": self.feature_error_thresholds[i] #plot_values["threshold"].reshape(-1,),
                }
            )

            data_type = 'Train data' if nr == 1 else 'Test data'
            y_layout = {
                "title": f"{data_type} | Forecast & reconstruction vs true value for channel {i}: {self.pred_cols[i] if self.pred_cols is not None else ''} ",
                "shapes": y_shapes,
                "yaxis": dict(range=[y_min, y_max]),
                "showlegend": True,
                #'template': 'simple_white'
            }

            e_layout = {
                "title": f"{data_type} | Total error for all channels" if show_tot_err else f"{data_type} | Error for channel {i}: {self.pred_cols[i] if self.pred_cols is not None else ''}",
                "shapes": e_shapes,
                "yaxis": dict(range=[0, e_max]),
                #'template': 'simple_white'
            }

            lines = [
                go.Scatter(x=y_df["y_true"].index, y=y_df["y_true"], line_color="rgb(0, 204, 150, 0.5)", name="y_true", line=dict(width=2,)),
                go.Scatter(x=y_df["y_forecast"].index, y=y_df["y_forecast"], line_color="rgb(255, 127, 14, 1)", name="y_forecast", line=dict(width=2,)),
                go.Scatter(x=y_df["y_recon"].index, y=y_df["y_recon"], line_color="rgb(31, 119, 180, 1)", name="y_recon", line=dict(width=2,)),
            ]

            fig = go.Figure(data=lines, layout=y_layout)
            py.offline.iplot(fig)

            e_lines = [
                go.Scatter(x=e_df['e_s'].index, y=e_df["e_s"], name="Error", line=dict(color='red', width=1,)),
                go.Scatter(x=e_df['threshold'].index, y=e_df["threshold"], name="Threshold", line=dict(color='black', width=1, dash='dash')),
            ]

            e_fig = go.Figure(data=e_lines, layout=e_layout)
            py.offline.iplot(e_fig)
            #py.offline.iplot(e_fig, kind='scatter', colors=["red", "black"], dash=[None, "dash"])

            #if plot_errors:
             #   e_df.iplot(kind="scatter", layout=e_layout, colors=["red", "black"], dash=[None, "dash"])

    def plot_all_channels(self, start=None, end=None, type="test"):
        if type == "train":
            data_copy = self.train_data.copy()
        elif type == "test":
            data_copy = self.test_data.copy()
        data_copy = data_copy.drop(columns=["Tot_A_Score", "threshold"])

        if start is not None and end is not None:
            assert start < end
        if start is not None:
            data_copy = data_copy.iloc[start:, :]
        if end is not None:
            start = 0 if start is None else start
            data_copy = data_copy.iloc[: end - start, :]

        num_cols = data_copy.shape[1]
        plt.tight_layout()
        colors = ["gray", "gray", "gray", "r"] * (num_cols // 4) + ["b", "g"]
        data_copy.plot(subplots=True, figsize=(20, num_cols), ylim=(0, 1.1), style=colors)
        plt.show()

    def plot_anomaly_segments(self, start=None, end=None, type="test"):
        if type == "train":
            data_copy = self.train_data.copy()
        elif type == "test":
            data_copy = self.test_data.copy()

        fig = make_subplots(rows=len(self.pred_cols), cols=1, shared_xaxes=True)
        shapes = []
        annotations = []
        for i in range(len(self.pred_cols)):
            values = data_copy[f'True_{i}'].values

            channel_error_threshold = self.feature_error_thresholds[i]
            channel_anomaly_scores = data_copy[f"A_Score_{i}"].values
            anomaly_preds = (channel_anomaly_scores > channel_error_threshold).astype(int)
            anomaly_sequences = self.get_anomaly_sequences(anomaly_preds)

            y_min = values.min()
            y_max = values.max()
            y_min -= 0.1 * y_max
            y_max += 0.5 * y_max

            j = i+1
            xref = f'x{j}' if i > 0 else 'x'
            yref = f'y{j}' if i > 0 else 'y'
            anomaly_shape = self.create_shapes(anomaly_sequences, None, y_min, y_max, None, xref=xref, yref=yref)
            shapes.extend(anomaly_shape)

            fig.append_trace(
                go.Scatter(y=values, line=dict(color='gray', width=1)),
                row=i+1, col=1
            )

            annotations.append(dict(
                xref=xref, yref=yref, text=self.pred_cols[i],
                showarrow=False, #align='right'
            ))

        fig.update_layout(height=4000, width=1500, shapes=shapes, template='simple_white', annotations=annotations)
        fig.update_yaxes(showticklabels=False)
        py.offline.iplot(fig)

    def plot_errors(self, channel, type="test"):
        if type == "train":
            return
        elif type == "test":
            data_copy = self.test_data.copy()

        fig, axs = plt.subplots(
            2,
            figsize=(30, 10),
            sharex=True,
        )
        if channel == "all":
            axs[0].plot(data_copy[f"Tot_A_Score"], c="r", label="anomaly scores")
        else:
            axs[0].plot(data_copy[f"A_Score_{channel}"], c="r", label="anomaly scores")
        axs[0].plot(data_copy["threshold"], linestyle="dashed", c="black", label="threshold")
        axs[1].plot(data_copy["True_Anomaly"], label="actual anomalies", alpha=0.7)
        axs[0].set_ylim([0, 2 * data_copy["threshold"].mean()])
        fig.legend(prop={"size": 20})
        plt.show()
