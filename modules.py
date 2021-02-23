from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionLayer(nn.Module):
	"""
	Single Graph Spatial/Feature Attention Layer
	"""

	def __init__(self, num_nodes, window_size, dropout, alpha):
		super(SpatialAttentionLayer, self).__init__()
		self.num_nodes = num_nodes
		self.window_size = window_size
		self.dropout = dropout
		self.leakyrelu = nn.LeakyReLU(alpha)

		self.w = nn.Parameter(torch.empty((2 * window_size, 1)))
		nn.init.xavier_uniform_(self.w.data, gain=1.414)

	# self.a = nn.Linear(2 * out_dim, 1, bias=False)

	def forward(self, x):
		# x has shape (n, k) where n is window size and k is number of nodes
		# For spatial attention we represent each node by a sequential vector,
		# containing all its values within the window

		v = x.T
		#Wh = torch.mm(h, self.W)  # Transformation shared across nodes

		# Creating matrix of concatenations of node features
		attn_input = self._make_attention_input(v)

		# Attention scores
		e = self.leakyrelu(torch.matmul(attn_input, self.w).squeeze(2))

		# Attention weights
		attention = torch.softmax(e, dim=1)
		attention = torch.dropout(attention, self.dropout, train=self.training)

		# Computing new node features using the attention
		h = torch.matmul(attention, v)

		return h

	def _make_attention_input(self, v):
		""" Preparing the spatial attention mechanism.
			Creating matrix with all possible combinations of concatenations of node:
				v1 || v1,
				...
				v1 || vK,
				v2 || v1,
				...
				v2 || vK,
				...
				...
				vK || v1,
				...
				vK || vK,
		"""

		K = self.num_nodes
		Wh_blocks_repeating = v.repeat_interleave(K, dim=0)  # Left-side of the matrix
		Wh_blocks_alternating = v.repeat(K, 1)  # Right-side of the matrix

		combined = torch.cat((Wh_blocks_repeating, Wh_blocks_alternating), dim=1)  # Shape (K*K, 2*window_size)
		return combined.view(K, K, 2 * self.window_size)


class TemporalAttentionLayer(nn.Module):
	"""
	Single Graph Temporal Attention Layer
	"""

	def __init__(self, num_nodes, window_size, dropout, alpha):
		super(TemporalAttentionLayer, self).__init__()
		self.num_nodes = num_nodes
		self.window_size = window_size
		self.dropout = dropout
		self.leakyrelu = nn.LeakyReLU(alpha)

		self.w = nn.Parameter(torch.empty((2 * num_nodes, 1)))
		nn.init.xavier_uniform_(self.w.data, gain=1.414)

	# self.a = nn.Linear(2 * out_dim, 1, bias=False)

	def forward(self, x):
		# x has shape (b, n, k) where b is batch size, n is window size and k is number of nodes
		# For temporal attention each node attend to its previous values,

		# Creating matrix of concatenations of node features
		attn_input = self._make_attention_input(x)

		# Attention scores
		e = self.leakyrelu(torch.matmul(attn_input, self.w).squeeze(2))

		# Attention weights
		attention = torch.softmax(e, dim=1)
		attention = torch.dropout(attention, self.dropout, train=self.training)

		# Computing new node features using the attention
		h = torch.matmul(attention, x)

		return h

	def _make_attention_input(self, x):
		""" Preparing the temporal attention mechanism.
			Creating matrix with all possible combinations of concatenations of node values:
				v1_tN || v1_t1,
				...
				v1_tN || v1_tN,
				v2_tN || v2_t1,
				...
				v2_tN || v1_tN,
				...
				...
				vK_tN || vK_t1,
				...
				vK_tN || vK_tN,
		"""

		# x has shape (b, n, k)
		v=x

		K = self.window_size
		Wh_blocks_repeating = v.repeat_interleave(K, dim=0)  # Left-side of the matrix
		Wh_blocks_alternating = v.repeat(K, 1)  # Right-side of the matrix

		combined = torch.cat((Wh_blocks_repeating, Wh_blocks_alternating), dim=1)  # Shape (K*K, 2*window_size)
		return combined.view(K, K, 2 * self.num_nodes)


class GRU(nn.Module):
	"""
	Gated Recurrent Unit (GRU)
	"""

	def __init__(self, in_dim, hid_dim, n_layers, dropout, device='cpu'):
		super(GRU, self).__init__()
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		self.device = device

		self.gru = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)
		self.relu = nn.ReLU()
		# self.hidden = None

	def forward(self, x, h):
		out, h = self.gru(x, h)
		return out, h

	def init_hidden(self, batch_size):
		# weight = next(self.parameters()).data
		# hidden = weight.new(self.n_layers, batch_size, self.hid_dim).zero_().to(self.device)
		return torch.zeros(self.n_layers, batch_size, self.hid_dim).to(self.device)

class Forecasting_Model(nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout, device='cpu'):
		super(Forecasting_Model, self).__init__()

		layers = []
		layers.append(nn.Linear(in_dim, hid_dim))
		for _ in range(n_layers - 1):
			layers.append(nn.Linear(hid_dim, hid_dim))

		layers.append(nn.Linear(hid_dim, out_dim))

		self.layers = nn.ModuleList(layers)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()


	def forward(self, x):
		for i in range(len(self.layers)-1):
			x = self.relu(self.layers[i](x))
			x = self.dropout(x)
		return self.layers[-1](x)


# window_size= 10
# num_nodes = 8
#
# att_out_dim = 3 * num_nodes
#
# x = torch.ones((window_size, att_out_dim))
# print(f'x: {x.shape}')
#
# in_dim = att_out_dim
# gru_hid_dim = 16
# horizon = 1
#
# gru = GRU(in_dim, gru_hid_dim, out_dim=horizon, n_layers=3, dropout=0.2)
#
# h_init = gru.init_hidden(window_size)
# out, h = gru(x.unsqueeze(1), h_init)
#
# print(f'out: {out.shape}')
# print(f'h: {h.shape}')
#
# fc_hid_dim = 32
# forecasting_model = Forecasting_Model(in_dim=gru_hid_dim, hid_dim=fc_hid_dim, horizon=horizon, n_layers=3, dropout=0.2)
#
# predictions = forecasting_model(out)
# print(predictions.shape)


# spat_gat_layer = SpatialAttentionLayer(num_nodes, window_size, dropout=0.1, alpha=0.2)
# out1 = spat_gat_layer(x)
# print(out1.shape)
#
# temp_gat_layer = TemporalAttentionLayer(num_nodes, window_size, dropout=0.1, alpha=0.2)
# out2 = temp_gat_layer(x)
# print(out2.shape)
#
# print('-'*10)
# out = torch.cat([out1.T, out2], dim=1)
# print(out.shape)