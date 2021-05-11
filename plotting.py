import pandas as pd
import numpy as np
import os
import json
import plotly as py
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import cufflinks as cf

from utils import get_data_dim
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
        self.train_output = None
        self.test_output = None
        self._load_results()
        self.labels_available = False
        self.pred_cols = None
        self.feature_error_thresholds = []
        self.start_datetime = None
        self.end_datetime = None
        self.lookback = None
        self.use_timestamps = False

        if "TELENOR" in result_path:
            path ='../datasets/telenor/site_data/vol2/metadata.txt'
            with open(path) as f:
                metadata = json.load(f)
                input_cols = np.array(metadata["columns"][2:])
                self.start_datetime = metadata['start']
                self.end_datetime = metadata['end']

            self.pred_cols = input_cols
            config_path = f'{result_path}/config.txt'
            with open(config_path) as f:
               self.lookback = json.load(f)['lookback']

            # Update data with datetimes
            self.use_timestamps = True
            num_train_points = len(self.train_output) if self.train_output is not None else None
            num_test_points = len(self.test_output) if self.test_output is not None else None
            all_timestamps = pd.date_range(self.start_datetime, self.end_datetime, freq="H").tolist()
            train_timestamps = all_timestamps[self.lookback:self.lookback+num_train_points]
            test_timestamps = all_timestamps[-num_test_points:]

            self.train_timestamps = train_timestamps
            self.test_timestamps = test_timestamps
            self.train_output['timestamp'] = train_timestamps
            self.test_output['timestamp'] = test_timestamps
            self.train_end = train_timestamps[-1]
            self.test_start = test_timestamps[0]

            # TODO: threshold now saved as part of output df, load them from there instead of calculating them again
            # Make column-wise anomaly predictions (based on column-wise thresholding)
            for i in range(len(self.pred_cols)):
                feature_anomaly_scores = self.train_output[f'A_Score_{i}'].values
                epsilon = find_epsilon(feature_anomaly_scores, reg_level=0)
                self.feature_error_thresholds.append(epsilon)

        else:
            self.labels_available = True
            self.train_output['timestamp'] = self.train_output.index
            self.test_output['timestamp'] = self.test_output.index
            if 'SMD' in result_path:
                self.pred_cols = [f'feat_{i}' for i in range(get_data_dim('machine'))]
            elif 'SMAP' in result_path or 'MSL' in result_path:
                self.pred_cols = ['feat_1']

            # TODO: threshold now saved as part of output df, load them from there instead of calculating them again
            for i in range(len(self.pred_cols)):
                feature_anomaly_scores = self.train_output[f'A_Score_{i}'].values
                epsilon = find_epsilon(feature_anomaly_scores)
                self.feature_error_thresholds.append(epsilon)

    def _load_results(self):
        print(f"Loading results of {self.result_path}")
        train_output = pd.read_pickle(f"{self.result_path}/train_output.pkl")
        train_anomaly_scores = np.load(f"{self.result_path}/train_scores.npy")
        train_output["A_Score_Global"] = train_anomaly_scores
        train_output["A_True_Global"] = 0

        test_output = pd.read_pickle(f"{self.result_path}/test_output.pkl")
        test_anomaly_scores = np.load(f"{self.result_path}/test_scores.npy")
        test_output["A_Score_Global"] = test_anomaly_scores

        self.train_output = train_output
        self.test_output = test_output

    def result_summary(self):
        path = f"{self.result_path}/summary.txt"
        if not os.path.exists(path):
            print(f"Folder {self.result_path} do not have a summary.txt file")
            return
        try:
            print("Result summary:")
            with open(path) as f:
                result_dict = json.load(f)["epsilon_results"]
                print(f"\tTP: {result_dict['TP']}")
                print(f"\tTN: {result_dict['TN']}")
                print(f"\tFP: {result_dict['FP']}")
                print(f"\tFN: {result_dict['FN']}")
                print(f"\tAvg latency: {result_dict['latency']:.2f}")
                print()
                print(f"\tPrecison:  {result_dict['precision']:.4f}")
                print(f"\tRecall:    {result_dict['recall']:.4f}")
                print(f"\tF1:        {result_dict['f1']:.4f}")
        except FileNotFoundError as e:
            print(e)
        except Exception:
            print('\tNo results because labels are not available')

    def create_shapes(self, ranges, sequence_type, _min, _max, plot_values, is_test=True, xref=None, yref=None):
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
            color = 'blue'
        else:
            color = "red" if sequence_type == "true" else "blue"
        shapes = []

        n_train = len(self.train_timestamps)
        n_test = len(self.test_timestamps)
        for r in ranges:
            w = 5
            if self.use_timestamps:
                x0 = self.test_timestamps[max(0, r[0] - w)] if is_test else self.train_timestamps[max(0, r[0] - w)]
                x1 = self.test_timestamps[min(n_test-1, r[1] + w)] if is_test else self.train_timestamps[min(n_train-1, r[1] + w)]
            else:
                x0 = r[0] - w
                x1 = r[1] + w
            shape = {
                "type": "rect",
                "x0": x0,  # self.config.l_s,
                "y0": _min,
                "x1": x1,  # self.config.l_s,
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

    def plot_channel(self, channel, plot_train=False, show_tot_err=False, start=None, end=None, plot_errors=True):
        """Plot forecasting, reconstruction, true value of a specific channel (feature),
        along with the anomaly score for that channel
        """

        test_copy = self.test_output.copy()

        if start is not None and end is not None:
            assert start < end
        if start is not None:
            test_copy = test_copy.iloc[start:, :]
        if end is not None:
            start = 0 if start is None else start
            test_copy = test_copy.iloc[: end - start, :]

        plot_data = [test_copy]

        if plot_train:
            train_copy = self.train_output.copy()
            plot_data.append(train_copy)

        for nr, data_copy in enumerate(plot_data):
            is_test = nr == 0

            if channel < 0 or f"Forecast_{channel}" not in data_copy.columns:
                raise Exception(f"Channel {channel} not present in data.")

            i = channel
            plot_values = {
                "timestamp": data_copy['timestamp'].values,
                "y_forecast": data_copy[f"Forecast_{i}"].values,
                "y_recon": data_copy[f"Recon_{i}"].values,
                "y_true": data_copy[f"True_{i}"].values,
                "errors": data_copy[f"A_Score_Global"].values if show_tot_err else data_copy[f"A_Score_{i}"].values,
                # "threshold": self.feature_error_thresholds[i] if not show_tot_err else data_copy["Thresh_Global"].values,
                "threshold": (data_copy[f'Thresh_{i}'] if f'Thresh{i}' in data_copy.columns else self.feature_error_thresholds[i])
                if not show_tot_err else data_copy["Thresh_Global"].values
            }

            anomaly_sequences = {
                # 'pred': self.get_anomaly_sequences(anomaly_preds),
                "pred": self.get_anomaly_sequences(data_copy[f"A_Pred_{i}"].values),
                "true": self.get_anomaly_sequences(data_copy["A_True_Global"].values),
            }

            if "TELENOR" in self.result_path:
                y_min = plot_values["y_true"].min()
                y_max = plot_values["y_true"].max()
                e_max = 2
                y_min -= 0.1 * y_max
                y_max += 0.5 * y_max
                y_min = -0.1
            else:
                y_min = 1.1 * plot_values["y_true"].min()
                y_max = 1.1 * plot_values["y_true"].max()
                e_max = 1.5 * plot_values["errors"].max()

            # y_shapes = create_shapes(segments, 'true', y_min, y_max, plot_values)
            if self.labels_available:
                y_shapes = self.create_shapes(anomaly_sequences["true"], "true", y_min, y_max, plot_values, is_test=is_test)
                e_shapes = self.create_shapes(anomaly_sequences["true"], "true", 0, e_max, plot_values, is_test=is_test)

                y_shapes += self.create_shapes(anomaly_sequences["pred"], "predicted", y_min, y_max, plot_values, is_test=is_test)
                e_shapes += self.create_shapes(anomaly_sequences["pred"], "predicted", 0, e_max, plot_values, is_test=is_test)
            else:
                y_shapes = self.create_shapes(anomaly_sequences["pred"], None, y_min, y_max, plot_values, is_test=is_test)
                e_shapes = self.create_shapes(anomaly_sequences["pred"], None, 0, e_max, plot_values, is_test=is_test)

            y_df = pd.DataFrame(
                {
                    "timestamp": plot_values["timestamp"].reshape(-1, ),
                    "y_forecast": plot_values["y_forecast"].reshape(-1,),
                    "y_recon": plot_values["y_recon"].reshape(-1,),
                    "y_true": plot_values["y_true"].reshape(-1,),
                }
            )

            e_df = pd.DataFrame(
                {
                    "timestamp": plot_values['timestamp'],
                    "e_s": plot_values["errors"].reshape(-1,),
                    "threshold": plot_values['threshold'],
                }
            )

            data_type = 'Test data' if is_test else 'Train data'
            y_layout = {
                "title": f"{data_type} | Forecast & reconstruction vs true value for channel {i}: {self.pred_cols[i] if self.pred_cols is not None else ''} ",
                "shapes": y_shapes,
                "yaxis": dict(range=[y_min, y_max]),
                "showlegend": True,
                'height': 400,
                'width': 1100,
                #'template': 'simple_white'
            }

            e_layout = {
                "title": f"{data_type} | Total error for all channels" if show_tot_err else f"{data_type} | Error for channel {i}: {self.pred_cols[i] if self.pred_cols is not None else ''}",
                "shapes": e_shapes,
                "yaxis": dict(range=[0, e_max]),
                'height': 400,
                'width': 1100,
                #'template': 'simple_white'
            }

            lines = [
                go.Scatter(x=y_df['timestamp'], y=y_df["y_true"], line_color="rgb(0, 204, 150, 0.5)", name="y_true", line=dict(width=2,)),
                go.Scatter(x=y_df['timestamp'], y=y_df["y_forecast"], line_color="rgb(255, 127, 14, 1)", name="y_forecast", line=dict(width=2,)),
                go.Scatter(x=y_df['timestamp'], y=y_df["y_recon"], line_color="rgb(31, 119, 180, 1)", name="y_recon", line=dict(width=2,)),
            ]

            fig = go.Figure(data=lines, layout=y_layout)
            py.offline.iplot(fig)

            e_lines = [
                go.Scatter(x=e_df['timestamp'], y=e_df["e_s"], name="Error", line=dict(color='red', width=1,)),
                go.Scatter(x=e_df['timestamp'], y=e_df["threshold"], name="Threshold", line=dict(color='black', width=1, dash='dash')),
            ]

            e_fig = go.Figure(data=e_lines, layout=e_layout)
            py.offline.iplot(e_fig)


    def plot_all_channels(self, start=None, end=None, type="test"):
        if type == "train":
            data_copy = self.train_output.copy()
        elif type == "test":
            data_copy = self.test_output.copy()
        data_copy = data_copy.drop(columns=["A_Score_Global", "Thresh_Global"])

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

    def plot_anomaly_segments(self, type="test", num_aligned_segments=None):
        is_test = True
        if type == "train":
            data_copy = self.train_output.copy()
            is_test = False
        elif type == "test":
            data_copy = self.test_output.copy()

        fig = make_subplots(rows=len(self.pred_cols), cols=1, shared_xaxes=True)
        if self.use_timestamps:
            timestamps = self.test_timestamps if is_test else self.train_timestamps
        else:
            timestamps = None
        shapes = []
        annotations = []
        for i in range(len(self.pred_cols)):
            values = data_copy[f'True_{i}'].values
            anomaly_sequences = self.get_anomaly_sequences(data_copy[f"A_Pred_{i}"].values)

            y_min = -0.1
            y_max = 2 #0.5 * y_max

            j = i + 1
            xref = f'x{j}' if i > 0 else 'x'
            yref = f'y{j}' if i > 0 else 'y'
            anomaly_shape = self.create_shapes(anomaly_sequences, None, y_min, y_max, None, xref=xref, yref=yref, is_test=is_test)
            shapes.extend(anomaly_shape)

            fig.append_trace(
                go.Scatter(x=timestamps, y=values, line=dict(color='black', width=1)),
                row=i+1, col=1
            )
            fig.update_yaxes(range=[-0.1, 1.5], row=i+1, col=1)

            annotations.append(dict(
                xref=xref, yref=yref, text=self.pred_cols[i],
                font=dict(size=10),
                showarrow=False, align='right', valign='top'
            ))

        # colors = ['aliceblue', 'aqua', 'aquamarine', 'azure', 'bisque', 'black', 'blue',
        #         'blueviolet', 'brown', 'burlywood', 'cadetblue',
        #         'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        #         'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
        #         'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
        #         'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
        #         'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        #         'darkslateblue', 'darkslategray', 'darkslategrey',
        #         'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        #         'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
        #         'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
        #         'ghostwhite', 'gold', 'goldenrod', 'gray', 'green',
        #         'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo']
        # colors = ['blue', 'aquamarine', 'green', 'aliceblue', 'black', 'brown', 'orange', 'aqua',  'darkturquoise']
        colors = ['blue', 'green', 'red', 'black', 'orange', 'brown', 'aqua', 'hotpink']
        taken_shapes_i = []
        keep_segments_i = []
        corr_segments_count = 0
        for nr, i in enumerate(range(len(shapes))):
            corr_shapes = [i]
            shape = shapes[i]
            shape['opacity'] = 0.3
            shape_x = shape['x0']

            for j in range(i+1, len(shapes)):
                if j not in taken_shapes_i and shapes[j]['x0'] == shape_x:
                    corr_shapes.append(j)

            if num_aligned_segments is not None:
                if num_aligned_segments[0] == '>':
                    num = int(num_aligned_segments[1:])
                    keep_segment = len(corr_shapes) >= num
                else:
                    num = int(num_aligned_segments)
                    keep_segment = len(corr_shapes) == num

                if keep_segment:
                    keep_segments_i.extend(corr_shapes)
                    taken_shapes_i.extend(corr_shapes)
                    if len(corr_shapes) != 1:
                        for shape_i in corr_shapes:
                            shapes[shape_i]['fillcolor'] = colors[corr_segments_count % len(colors)]
                        corr_segments_count += 1

        if num_aligned_segments is not None:
            shapes = np.array(shapes)
            shapes = shapes[keep_segments_i].tolist()

        fig.update_layout(height=2000, width=1200, shapes=shapes, template='simple_white',
                          annotations=annotations, showlegend=False,
                          xaxis_range=[self.start_datetime if not is_test else self.test_start,
                                       self.train_end if not is_test else self.end_datetime] if self.use_timestamps else None)
        fig.update_yaxes(ticks="", showticklabels=False, showline=True, mirror=True)
        fig.update_xaxes(ticks="", showticklabels=False, showline=True, mirror=True)
        py.offline.iplot(fig)

    def plot_global_predictions(self, type="test"):
        if type == "test":
            data_copy = self.test_output.copy()
        else:
            data_copy = self.train_output.copy()

        fig, axs = plt.subplots(
            3,
            figsize=(30, 10),
            sharex=True,
        )
        axs[0].plot(data_copy[f"A_Score_Global"], c="r", label="anomaly scores")
        axs[0].plot(data_copy["Thresh_Global"], linestyle="dashed", c="black", label="threshold")
        axs[1].plot(data_copy["A_Pred_Global"], label="predicted anomalies", c="orange")
        if self.labels_available and type == "test":
            axs[2].plot(data_copy["A_True_Global"], label="actual anomalies",)
        axs[0].set_ylim([None, 2 * data_copy["Thresh_Global"].mean()])
        fig.legend(prop={"size": 20})
        plt.show()

    def plotly_global_predictions(self, type="test"):
        is_test = True
        if type == "train":
            data_copy = self.train_output.copy()
            is_test = False
        elif type == "test":
            data_copy = self.test_output.copy()

        tot_anomaly_scores = data_copy["A_Score_Global"].values
        pred_anomaly_sequences = self.get_anomaly_sequences(data_copy[f"A_Pred_Global"].values)
        y_min = -0.1
        y_max = 1.1 * np.max(tot_anomaly_scores)
        shapes = self.create_shapes(pred_anomaly_sequences, 'pred', y_min, y_max, None, is_test=is_test)
        if self.labels_available and is_test:
            true_anomaly_sequences = self.get_anomaly_sequences(data_copy[f"A_True_Global"].values)
            shapes2 = self.create_shapes(true_anomaly_sequences, 'true', y_min, y_max, None, is_test=is_test)
            shapes.extend(shapes2)

        layout = {
            "title": f"{type} | Total error, predicted (blue) and true anomalies (red if available)",
            "shapes": shapes,
            "yaxis": dict(range=[0, y_max]),
            'height': 400,
            'width': 1500,
            # 'template': 'simple_white'
        }

        fig = go.Figure(data=[go.Scatter(x=data_copy['timestamp'], y=tot_anomaly_scores, line=dict(width=1, color='red'))], layout=layout)
        py.offline.iplot(fig)