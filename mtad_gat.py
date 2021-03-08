import torch
import torch.nn as nn

from modules import ConvLayer, FeatureAttentionLayer, TemporalAttentionLayer, GRU, Forecasting_Model, RNNAutoencoder


class MTAD_GAT(nn.Module):
	def __init__(self,  num_nodes, window_size, horizon, out_dim, batch_size,
				 kernel_size=7,
				 gru_n_layers=3,
				 gru_hid_dim=64,
				 autoencoder_n_layers=1,
				 autoencoder_hid_dim=128,
				 forecasting_n_layers=3,
				 forecasting_hid_dim=32,
				 dropout=0.2,
				 alpha=0.2,
				 device='cpu'):
		super(MTAD_GAT, self).__init__()
		self.conv = ConvLayer(num_nodes, window_size, kernel_size, device)
		self.feature_gat = FeatureAttentionLayer(num_nodes, window_size, dropout, alpha, device)
		self.temporal_gat = TemporalAttentionLayer(num_nodes, window_size, dropout, alpha, device)
		self.gru = GRU(3*num_nodes, gru_hid_dim, gru_n_layers, dropout, device)
		self.forecasting_model = Forecasting_Model(gru_hid_dim, forecasting_hid_dim, out_dim*horizon, forecasting_n_layers, dropout, device)
		self.recon_model = RNNAutoencoder(window_size, num_nodes, autoencoder_n_layers, autoencoder_hid_dim, num_nodes, dropout, device)

	def forward(self, x):
		# x shape (b, n, k): b - batch size, n - window size, k - number of nodes/features
		x = self.conv(x)

		h_feat = self.feature_gat(x)
		h_temp = self.temporal_gat(x)

		#h_cat = x
		#h_cat = torch.cat([x, h_temp], dim=2)
		h_cat = torch.cat([x, h_feat.permute(0, 2, 1), h_temp], dim=2) # (b, n, 3k)

		gru_out, last_hidden = self.gru(h_cat)

		forecasting_in = last_hidden.view(x.shape[0], -1) #gru_out[:, -1, :]  # Hidden state for to last timestamp

		# Flatten
		predictions = self.forecasting_model(forecasting_in)

		# TODO: Reconstruction model
		#recons = None
		recons = self.recon_model(h_cat)

		return predictions, recons

