import pandas as pd
import numpy as np
import os
import json
import plotly as py
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()


class Plotter:
	"""
	Class for visualizing results of anomaly detection.
	Includes visualization of forecasts, reconstructions, anomaly scores, predicted and actual anomalies
	"""
	def __init__(self, result_path):
		self.result_path = result_path
		self.data_name = self.result_path.split('/')[-1]
		self.data = None
		self.labels_available = False
		self._load_results()

	def _load_results(self):
		print(f"Loading results of {self.result_path}")
		anomaly_preds = pd.read_pickle(f'{self.result_path}/anomaly_preds.pkl')
		if not self.data_name == "TELENOR":
			self.labels_available = True
			anomaly_preds['anomaly'] = anomaly_preds['anomaly'].astype(int)
		output = pd.read_pickle(f'{self.result_path}/preds.pkl')
		test_anomaly_scores = np.load(f"{self.result_path}/test_scores.npy")

		output['Tot_A_Score'] = test_anomaly_scores
		output['Pred_Anomaly'] = anomaly_preds['pred_anomaly']
		output['True_Anomaly'] = anomaly_preds['anomaly']
		output['threshold'] = anomaly_preds['threshold']
		self.data = output

	def result_summary(self):
		if not self.labels_available:
			print(f'Labels not available.')
			return

		path = f'{self.result_path}/summary.txt'
		if not os.path.exists(path):
			print(f'Folder {self.result_path} do not have a summary.txt file')
		try:
			print('Result summary:')
			with open(path) as f:
				result_dict = json.load(f)['pot_result']
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

	def create_shapes(self, ranges, sequence_type, _min, _max, plot_values):
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

		if not _max:
			_max = max(plot_values['errors'])

		color = 'red' if sequence_type == 'true' else 'blue'
		shapes = []

		for r in ranges:
			shape = {
				'type': 'rect',
				'x0': r[0] - 3,  # self.config.l_s,
				'y0': _min,
				'x1': r[1] + 2,  # self.config.l_s,
				'y1': _max,
				'fillcolor': color,
				'opacity': 0.2,
				'line': {
					'width': 0,
				}
			}
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

	def plot_channel(self, channel, show_tot_err=False, start=None, end=None, plot_errors=True):
		""" Plot forecasting, reconstruction, true value of a specific channel (feature),
			along with the anomaly score for that channel
		"""
		if channel < 0 or f'Pred_{channel}' not in self.data.columns:
			raise Exception(f'Channel {channel} not present in data.')

		data_copy = self.data.copy()
		if start is not None and end is not None:
			assert (start < end)
		if start is not None:
			data_copy = data_copy.iloc[start:, :]
		if end is not None:
			start = 0 if start is None else start
			data_copy = data_copy.iloc[:end - start, :]

		i = channel
		plot_values = {
			'y_forecast': data_copy[f'Pred_{i}'].values,
			'y_recon': data_copy[f'Recon_{i}'].values,
			'y_true': data_copy[f'True_{i}'].values,
			# 'errors': output_copy[f'A_Score_{i}'].values,
			'errors': data_copy[f'Tot_A_Score'].values if show_tot_err else data_copy[f'A_Score_{i}'].values,
			'threshold': data_copy['threshold'].values,
		}

		anomaly_sequences = {
			'pred': self.get_anomaly_sequences(data_copy['Pred_Anomaly'].values),
			'true': self.get_anomaly_sequences(data_copy['True_Anomaly'].values)
		}

		y_min = plot_values['y_true'].min()
		y_max = plot_values['y_true'].max()
		e_max = plot_values['errors'].max()

		y_min -= 0.3 * y_max
		y_max += 0.5 * y_max
		e_max += 0.5 * e_max

		# y_shapes = create_shapes(segments, 'true', y_min, y_max, plot_values)
		y_shapes = self.create_shapes(anomaly_sequences['true'], 'true', y_min, y_max, plot_values)
		e_shapes = self.create_shapes(anomaly_sequences['true'], 'true', 0, e_max, plot_values)

		y_shapes += self.create_shapes(anomaly_sequences['pred'], 'predicted', y_min, y_max, plot_values)
		e_shapes += self.create_shapes(anomaly_sequences['pred'], 'predicted', 0, e_max, plot_values)

		y_df = pd.DataFrame({
			'y_forecast': plot_values['y_forecast'].reshape(-1, ),
			'y_recon': plot_values['y_recon'].reshape(-1, ),
			'y_true': plot_values['y_true'].reshape(-1, )
		})

		e_df = pd.DataFrame({
			'e_s': plot_values['errors'].reshape(-1, ),
			'threshold': plot_values['threshold'].reshape(-1, )
		})

		y_layout = {
			'title': f'Forecast & reconstruction vs true value for channel: {i} ',
			'shapes': y_shapes,
			'yaxis': dict(
				range=[y_min, y_max]
			),
			'showlegend': True
		}

		e_layout = {
			'title': "Total error for all channels" if show_tot_err else f"Error for channel: {i}",
			'shapes': e_shapes,
			'yaxis': dict(
				range=[0, e_max]
			)
		}

		lines = [
			go.Scatter(x=y_df['y_true'].index, y=y_df['y_true'], line_color='rgb(0, 204, 150, 0.5)', name='y_true'),
			go.Scatter(x=y_df['y_forecast'].index, y=y_df['y_forecast'], line_color='rgb(255, 127, 14, 1)',
					   name='y_forecast'),
			go.Scatter(x=y_df['y_recon'].index, y=y_df['y_recon'], line_color='rgb(31, 119, 180, 1)', name='y_recon'),
		]

		fig = go.Figure(
			data=lines,
			layout=y_layout
		)
		py.offline.iplot(fig)

		if plot_errors:
			e_df.iplot(kind='scatter', layout=e_layout, colors=['red', 'black'], dash=[None, 'dash'])

	def plot_all_channels(self, start=None, end=None):
		data_copy = self.data.copy().drop(columns=['Tot_A_Score', 'threshold'])
		if start is not None and end is not None:
			assert (start < end)
		if start is not None:
			data_copy = data_copy.iloc[start:, :]
		if end is not None:
			start = 0 if start is None else start
			data_copy = data_copy.iloc[:end - start, :]

		num_cols = data_copy.shape[1]
		plt.tight_layout()
		colors = ['gray', 'gray', 'gray', 'r'] * (num_cols // 4) + ['b', 'g']
		data_copy.plot(subplots=True, figsize=(20, num_cols), ylim=(0, 1.1), style=colors)
		plt.show()

	def plot_errors(self, channel):
		fig, axs = plt.subplots(2, figsize=(30, 10), sharex=True, )
		if channel == 'all':
			axs[0].plot(self.data[f'Tot_A_Score'], c='r', label='anomaly scores')
		else:
			axs[0].plot(self.data[f'A_Score_{channel}'], c='r', label='anomaly scores')
		axs[0].plot(self.data['threshold'], linestyle='dashed', c='black', label='threshold')
		axs[1].plot(self.data['True_Anomaly'], label='actual anomalies', alpha=0.7)
		axs[0].set_ylim([0, 2*self.data['threshold'].mean()])
		fig.legend(prop={'size': 20})
		plt.show()
