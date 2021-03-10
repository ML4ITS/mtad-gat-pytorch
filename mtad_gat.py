import torch
import torch.nn as nn

from modules import ConvLayer, FeatureAttentionLayer, TemporalAttentionLayer, GRU, Forecasting_Model, RNNAutoencoder


class MTAD_GAT(nn.Module):
	def __init__(self,  n_features, window_size, horizon, out_dim, batch_size,
				 kernel_size=7,
				 gru_n_layers=3,
				 gru_hid_dim=64,
				 autoenc_n_layers=1,
				 autoenc_hid_dim=128,
				 forecast_n_layers=3,
				 forecast_hid_dim=32,
				 dropout=0.2,
				 alpha=0.2,
				 use_cuda=True):
		super(MTAD_GAT, self).__init__()

		device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

		self.conv = ConvLayer(n_features, window_size, kernel_size, device)
		self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, device)
		self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, device)
		self.gru = GRU(2*n_features, gru_hid_dim, gru_n_layers, dropout, device)
		self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim*horizon, forecast_n_layers, dropout, device)
		self.recon_model = RNNAutoencoder(window_size, gru_hid_dim, autoenc_n_layers, autoenc_hid_dim, n_features, dropout, device)

		self.model_args = f'MTAD-GAT(k_size={kernel_size}, gru_n_layers={gru_n_layers}, gru_hid_dim={gru_hid_dim}, ' \
						   f'autoencoder_n_layers={autoenc_n_layers}, autoencoder_hid_dim={autoenc_hid_dim}, ' \
						   f'forecast_n_layers={forecast_n_layers}, forecast_hid_dim={forecast_hid_dim}, ' \
						   f'dropout={dropout}, alpha={alpha})'
	def __repr__(self):
		return self.model_args

	def forward(self, x):
		# x shape (b, n, k): b - batch size, n - window size, k - number of nodes/features
		x = self.conv(x)

		h_feat = self.feature_gat(x)
		#h_temp = self.temporal_gat(x)

		#h_cat = torch.cat([x, h_feat.permute(0, 2, 1), h_temp], dim=2) # (b, n, 3k)

		h_cat = torch.cat([x, h_feat.permute(0, 2, 1)], dim=2)

		gru_out, h_end = self.gru(h_cat)  # (

		h_end = h_end.view(x.shape[0], -1) #gru_out[:, -1, :]  # Hidden state for last timestamp

		predictions = self.forecasting_model(h_end)
		recons = self.recon_model(h_end)

		return predictions, recons

