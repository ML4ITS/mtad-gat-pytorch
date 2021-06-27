from utils import get_data_dim, get_series_color, get_y_height
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import plotly as py
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import cufflinks as cf
cf.go_offline()


class Plotter:

    """
    Class for visualizing results of anomaly detection.
    Includes visualization of forecasts, reconstructions, anomaly scores, predicted and actual anomalies
    Plotter-class inspired by TelemAnom (https://github.com/khundman/telemanom)
    """

    def __init__(self, result_path, model_id='-1'):
        self.result_path = result_path
        self.model_id = model_id
        self.train_output = None
        self.test_output = None
        self.labels_available = True
        self.pred_cols = None
        self._load_results()
        self.train_output["timestamp"] = self.train_output.index
        self.test_output["timestamp"] = self.test_output.index

        config_path = f"{self.result_path}/config.txt"
        with open(config_path) as f:
            self.lookback = json.load(f)["lookback"]

        if "SMD" in self.result_path:
            self.pred_cols = [f"feat_{i}" for i in range(get_data_dim("machine"))]
        elif "SMAP" in self.result_path or "MSL" in self.result_path:
            self.pred_cols = ["feat_1"]

    def _load_results(self):
        if self.model_id.startswith('-'):
            dir_content = os.listdir(self.result_path)
            datetimes = [datetime.strptime(subf, '%d%m%Y_%H%M%S') for subf in dir_content if os.path.isdir(f"{self.result_path}/{subf}")
                          and subf not in ['logs']]
            datetimes.sort()
            model_id = datetimes[int(self.model_id)].strftime('%d%m%Y_%H%M%S')
            self.result_path = f'{self.result_path}/{model_id}'

        print(f"Loading results of {self.result_path}")
        train_output = pd.read_pickle(f"{self.result_path}/train_output.pkl")
        train_output.to_pickle(f"{self.result_path}/train_output.pkl")
        train_output["A_True_Global"] = 0
        test_output = pd.read_pickle(f"{self.result_path}/test_output.pkl")

        # Because for SMAP and MSL only one feature is predicted
        if 'SMAP' in self.result_path or 'MSL' in self.result_path:
            train_output[f'A_Pred_0'] = train_output['A_Pred_Global']
            train_output[f'A_Score_0'] = train_output['A_Score_Global']
            train_output[f'Thresh_0'] = train_output['Thresh_Global']

            test_output[f'A_Pred_0'] = test_output['A_Pred_Global']
            test_output[f'A_Score_0'] = test_output['A_Score_Global']
            test_output[f'Thresh_0'] = test_output['Thresh_Global']

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
                result_dict = json.load(f)
                epsilon_result = result_dict["epsilon_result"]
                pot_result = result_dict["pot_result"]
                bf_results = result_dict["bf_result"]
                print(f'Epsilon:')
                print(f'\t\tprecision: {epsilon_result["precision"]:.2f}, recall: {epsilon_result["recall"]:.2f}, F1: {epsilon_result["f1"]:.2f}')
                print(f'POT:')
                print(f'\t\tprecision: {pot_result["precision"]:.2f}, recall: {pot_result["recall"]:.2f}, F1: {pot_result["f1"]:.2f}')
                print(f'Brute-Force:')
                print(f'\t\tprecision: {bf_results["precision"]:.2f}, recall: {bf_results["recall"]:.2f}, F1: {bf_results["f1"]:.2f}')

        except FileNotFoundError as e:
            print(e)

    def create_shapes(self, ranges, sequence_type, _min, _max, plot_values, is_test=True, xref=None, yref=None):
        """
        Create shapes for regions to highlight in plotly (true and predicted anomaly sequences).

        :param ranges: tuple of start and end indices for anomaly sequences for a feature
        :param sequence_type: "predict" if predicted values else "true" if actual values. Determines colors.
        :param _min: min y value of series
        :param _max: max y value of series
        :param plot_values: dictionary of different series to be plotted

        :return: list of shapes specifications for plotly
        """

        if _max is None:
            _max = max(plot_values["errors"])

        if sequence_type is None:
            color = "blue"
        else:
            color = "red" if sequence_type == "true" else "blue"
        shapes = []

        for r in ranges:
            w = 5
            x0 = r[0] - w
            x1 = r[1] + w
            shape = {
                "type": "rect",
                "x0": x0,
                "y0": _min,
                "x1": x1,
                "y1": _max,
                "fillcolor": color,
                "opacity": 0.08,
                "line": {
                    "width": 0,
                },
            }
            if xref is not None:
                shape["xref"] = xref
                shape["yref"] = yref

            shapes.append(shape)

        return shapes

    @staticmethod
    def get_anomaly_sequences(values):
        splits = np.where(values[1:] != values[:-1])[0] + 1
        if values[0] == 1:
            splits = np.insert(splits, 0, 0)

        a_seqs = []
        for i in range(0, len(splits) - 1, 2):
            a_seqs.append([splits[i], splits[i + 1] - 1])

        if len(splits) % 2 == 1:
            a_seqs.append([splits[-1], len(values) - 1])

        return a_seqs

    def plot_feature(self, feature, plot_train=False, plot_errors=True, plot_feature_anom=False, start=None, end=None):
        """
        Plot forecasting, reconstruction, true value of a specific feature (feature),
        along with the anomaly score for that feature
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

            if feature < 0 or f"Forecast_{feature}" not in data_copy.columns:
                raise Exception(f"Channel {feature} not present in data.")

            i = feature
            plot_values = {
                "timestamp": data_copy["timestamp"].values,
                "y_forecast": data_copy[f"Forecast_{i}"].values,
                "y_recon": data_copy[f"Recon_{i}"].values,
                "y_true": data_copy[f"True_{i}"].values,
                "errors": data_copy[f"A_Score_{i}"].values,
                "threshold": data_copy[f"Thresh_{i}"]
            }

            anomaly_sequences = {
                "pred": self.get_anomaly_sequences(data_copy[f"A_Pred_{i}"].values),
                "true": self.get_anomaly_sequences(data_copy["A_True_Global"].values),
            }

            if is_test and start is not None:
                anomaly_sequences['pred'] = [[s+start, e+start] for [s, e] in anomaly_sequences['pred']]
                anomaly_sequences['true'] = [[s+start, e+start] for [s, e] in anomaly_sequences['true']]

            y_min = 1.1 * plot_values["y_true"].min()
            y_max = 1.1 * plot_values["y_true"].max()
            e_max = 1.5 * plot_values["errors"].max()

            y_shapes = self.create_shapes(anomaly_sequences["pred"], "predicted", y_min, y_max, plot_values, is_test=is_test)
            e_shapes = self.create_shapes(anomaly_sequences["pred"], "predicted", 0, e_max, plot_values, is_test=is_test)
            if self.labels_available and ('SMAP' in self.result_path or 'MSL' in self.result_path):
                y_shapes += self.create_shapes(anomaly_sequences["true"], "true", y_min, y_max, plot_values, is_test=is_test)
                e_shapes += self.create_shapes(anomaly_sequences["true"], "true", 0, e_max, plot_values, is_test=is_test)

            y_df = pd.DataFrame(
                {
                    "timestamp": plot_values["timestamp"].reshape(-1,),
                    "y_forecast": plot_values["y_forecast"].reshape(-1,),
                    "y_recon": plot_values["y_recon"].reshape(-1,),
                    "y_true": plot_values["y_true"].reshape(-1,)
                }
            )

            e_df = pd.DataFrame(
                {
                    "timestamp": plot_values["timestamp"],
                    "e_s": plot_values["errors"].reshape(-1,),
                    "threshold": plot_values["threshold"],
                }
            )

            data_type = "Test data" if is_test else "Train data"
            y_layout = {
                "title": f"{data_type} | Forecast & reconstruction vs true value for {self.pred_cols[i] if self.pred_cols is not None else ''} ",
                "showlegend": True,
                "height": 400,
                "width": 1100,
            }

            e_layout = {
                "title": f"{data_type} | Error for {self.pred_cols[i] if self.pred_cols is not None else ''}",
                #"yaxis": dict(range=[0, e_max]),
                "height": 400,
                "width": 1100,
            }

            if plot_feature_anom:
                y_layout["shapes"] = y_shapes
                e_layout["shapes"] = e_shapes

            lines = [
                go.Scatter(
                    x=y_df["timestamp"],
                    y=y_df["y_true"],
                    line_color="rgb(0, 204, 150, 0.5)",
                    name="y_true",
                    line=dict(width=2)),
                go.Scatter(
                    x=y_df["timestamp"],
                    y=y_df["y_forecast"],
                    line_color="rgb(255, 127, 14, 1)",
                    name="y_forecast",
                    line=dict(width=2)),
                go.Scatter(
                    x=y_df["timestamp"],
                    y=y_df["y_recon"],
                    line_color="rgb(31, 119, 180, 1)",
                    name="y_recon",
                    line=dict(width=2)),
            ]

            fig = go.Figure(data=lines, layout=y_layout)
            py.offline.iplot(fig)

            e_lines = [
                go.Scatter(
                    x=e_df["timestamp"],
                    y=e_df["e_s"],
                    name="Error",
                    line=dict(color="red", width=1))]
            if plot_feature_anom:
                e_lines.append(
                    go.Scatter(
                        x=e_df["timestamp"],
                        y=e_df["threshold"],
                        name="Threshold",
                        line=dict(color="black", width=1, dash="dash")))

            if plot_errors:
                e_fig = go.Figure(data=e_lines, layout=e_layout)
                py.offline.iplot(e_fig)

    def plot_all_features(self, start=None, end=None, type="test"):
        """
        Plotting all features, using the following order:
            - forecasting for feature i
            - reconstruction for feature i
            - true value for feature i
            - anomaly score (error) for feature i
        """
        if type == "train":
            data_copy = self.train_output.copy()
        elif type == "test":
            data_copy = self.test_output.copy()

        data_copy = data_copy.drop(columns=['timestamp', 'A_Score_Global', 'Thresh_Global'])
        cols = [c for c in data_copy.columns if not (c.startswith('Thresh_') or c.startswith('A_Pred_'))]
        data_copy = data_copy[cols]

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
        data_copy.plot(subplots=True, figsize=(20, num_cols), ylim=(0, 1.5), style=colors)
        plt.show()

    def plot_anomaly_segments(self, type="test", num_aligned_segments=None, show_boring_series=False):
        """
        Finds collective anomalies, i.e. feature-wise anomalies that occur at the same time, and visualize them
        """
        is_test = True
        if type == "train":
            data_copy = self.train_output.copy()
            is_test = False
        elif type == "test":
            data_copy = self.test_output.copy()

        def get_pred_cols(df):
            pred_cols_to_remove = []
            col_names_to_remove = []
            for i, col in enumerate(self.pred_cols):
                y = df[f"True_{i}"].values
                if np.average(y) >= 0.95 or np.average(y) == 0.0:
                    pred_cols_to_remove.append(col)
                    cols = list(df.columns[4 * i : 4 * i + 4])
                    col_names_to_remove.extend(cols)

            df.drop(col_names_to_remove, axis=1, inplace=True)
            return [x for x in self.pred_cols if x not in pred_cols_to_remove]

        non_constant_pred_cols = self.pred_cols if show_boring_series else get_pred_cols(data_copy)

        fig = make_subplots(
            rows=len(non_constant_pred_cols),
            cols=1,
            vertical_spacing=0.4 / len(non_constant_pred_cols),
            shared_xaxes=True,
        )

        timestamps = None
        shapes = []
        annotations = []
        for i in range(len(non_constant_pred_cols)):
            new_idx = int(data_copy.columns[4 * i].split("_")[-1])
            values = data_copy[f"True_{new_idx}"].values

            anomaly_sequences = self.get_anomaly_sequences(data_copy[f"A_Pred_{new_idx}"].values)

            y_min = -0.1
            y_max = 2  # 0.5 * y_max

            j = i + 1
            xref = f"x{j}" if i > 0 else "x"
            yref = f"y{j}" if i > 0 else "y"
            anomaly_shape = self.create_shapes(
                anomaly_sequences, None, y_min, y_max, None, xref=xref, yref=yref, is_test=is_test
            )
            shapes.extend(anomaly_shape)

            fig.append_trace(
                go.Scatter(x=timestamps, y=values, line=dict(color=get_series_color(values), width=1)), row=i + 1, col=1
            )
            fig.update_yaxes(range=[-0.1, get_y_height(values)], row=i + 1, col=1)

            annotations.append(
                dict(
                    # xref="paper",
                    xanchor="left",
                    yref=yref,
                    text=f"<b>{non_constant_pred_cols[i].upper()}</b>",
                    font=dict(size=10),
                    showarrow=False,
                    yshift=35,
                    xshift=(-523),
                )
            )

        colors = ["blue", "green", "red", "black", "orange", "brown", "aqua", "hotpink"]
        taken_shapes_i = []
        keep_segments_i = []
        corr_segments_count = 0
        for nr, i in enumerate(range(len(shapes))):
            corr_shapes = [i]
            shape = shapes[i]
            shape["opacity"] = 0.3
            shape_x = shape["x0"]

            for j in range(i + 1, len(shapes)):
                if j not in taken_shapes_i and shapes[j]["x0"] == shape_x:
                    corr_shapes.append(j)

            if num_aligned_segments is not None:
                if num_aligned_segments[0] == ">":
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
                            shapes[shape_i]["fillcolor"] = colors[corr_segments_count % len(colors)]
                        corr_segments_count += 1

        if num_aligned_segments is not None:
            shapes = np.array(shapes)
            shapes = shapes[keep_segments_i].tolist()

        fig.update_layout(
            height=1800,
            width=1200,
            shapes=shapes,
            template="simple_white",
            annotations=annotations,
            showlegend=False)

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
            axs[2].plot(
                data_copy["A_True_Global"],
                label="actual anomalies",
            )
        axs[0].set_ylim([0, 5 * np.mean(data_copy["Thresh_Global"].values)])
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
        threshold = data_copy['Thresh_Global'].values
        y_min = -0.1
        y_max = 5 * np.mean(threshold) # np.max(tot_anomaly_scores)
        shapes = self.create_shapes(pred_anomaly_sequences, "pred", y_min, y_max, None, is_test=is_test)
        if self.labels_available and is_test:
            true_anomaly_sequences = self.get_anomaly_sequences(data_copy[f"A_True_Global"].values)
            shapes2 = self.create_shapes(true_anomaly_sequences, "true", y_min, y_max, None, is_test=is_test)
            shapes.extend(shapes2)

        layout = {
            "title": f"{type} set | Total error, predicted anomalies in blue, true anomalies in red if available "
                     f"(making correctly predicted in purple)",
            "shapes": shapes,
            "yaxis": dict(range=[0, y_max]),
            "height": 400,
            "width": 1500
        }

        fig = go.Figure(
            data=[go.Scatter(x=data_copy["timestamp"], y=tot_anomaly_scores, name='Error', line=dict(width=1, color="red")),
                  go.Scatter(x=data_copy["timestamp"], y=threshold, name='Threshold', line=dict(color="black", width=1, dash="dash"))],
            layout=layout,
        )
        py.offline.iplot(fig)
