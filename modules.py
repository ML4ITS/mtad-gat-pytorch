import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
	"""
	1-D Convolution layer to extract high-level features of each time-series input
	"""

	def __init__(self, num_nodes, window_size, kernel_size=7, device='cpu'):
		super(ConvLayer, self).__init__()
		self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
		self.conv = nn.Conv1d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=kernel_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = x.permute(0, 2, 1)  # To get the features/nodes as channel and timesteps as the spatial dimension
		x = self.padding(x)
		x = self.relu(self.conv(x))
		return x.permute(0, 2, 1)  # Permute back


class FeatureAttentionLayer(nn.Module):
	"""
	Single Graph Feature/Spatial Attention Layer
	"""

	def __init__(self, num_nodes, window_size, dropout, alpha, device='cpu'):
		super(FeatureAttentionLayer, self).__init__()
		self.num_nodes = num_nodes
		self.window_size = window_size
		self.dropout = dropout
		self.leakyrelu = nn.LeakyReLU(alpha)

		self.w = nn.Parameter(torch.empty((2 * window_size, 1)))
		nn.init.xavier_uniform_(self.w.data, gain=1.414)

	# self.a = nn.Linear(2 * out_dim, 1, bias=False)

	def forward(self, x):
		# x has shape (b, n, k) where n is window size and k is number of nodes
		# For spatial attention we represent each node by a sequential vector,
		# containing all its values within the window

		v = x.permute(0, 2, 1)

		#Wh = torch.mm(h, self.W)  # Transformation shared across nodes (not used)

		# Creating matrix of concatenations of node features
		attn_input = self._make_attention_input(v)

		# Attention scores
		e = self.leakyrelu(torch.matmul(attn_input, self.w)).squeeze(3)

		# Attention weights
		attention = torch.softmax(e, dim=2)
		attention = torch.dropout(attention, self.dropout, train=self.training)

		# Computing new node features using the attention
		h = torch.matmul(attention, v)

		return h

	def _make_attention_input(self, v):
		""" Preparing the feature attention mechanism.
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

		# print(f'v: {v.shape}')

		K = self.num_nodes
		Wh_blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
		Wh_blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix

		combined = torch.cat((Wh_blocks_repeating, Wh_blocks_alternating), dim=2)  # Shape (b, K*K, 2*window_size)

		return combined.view(v.size(0), K, K, 2 * self.window_size)


class TemporalAttentionLayer(nn.Module):
	"""
	Single Graph Temporal Attention Layer
	"""

	def __init__(self, num_nodes, window_size, dropout, alpha, device='cpu'):
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

		# print(f'x: {x.shape}')

		# Creating matrix of concatenations of node features
		attn_input = self._make_attention_input(x)

		# Attention scores
		e = self.leakyrelu(torch.matmul(attn_input, self.w).squeeze(3))

		# print(f'e: {e.shape}')

		# Attention weights
		attention = torch.softmax(e, dim=2)
		attention = torch.dropout(attention, self.dropout, train=self.training)

		# print(f'attention: {attention.shape}')

		# Computing new node features using the attention
		h = torch.matmul(attention, x)
		# print(f'h: {h.shape}')

		return h



	def _make_attention_input(self, v):
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

		# v has shape (b, n, k)
		# print(v[0])

		K = self.window_size
		Wh_blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
		Wh_blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix

		# print(Wh_blocks_repeating.shape)
		# print(Wh_blocks_alternating.shape)

		combined = torch.cat((Wh_blocks_repeating, Wh_blocks_alternating), dim=2)  # Shape (b, K*K, 2*window_size)

		# print(combined.shape)

		return combined.view(v.size(0), K, K, 2 * self.num_nodes)


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

	def forward(self, x):
		h0 = torch.zeros(self.n_layers, x.shape[0], self.hid_dim).to(self.device)
		out, h = self.gru(x, h0)
		return out, h

	def init_hidden(self, batch_size):
		# weight = next(self.parameters()).data
		# hidden = weight.new(self.n_layers, batch_size, self.hid_dim).zero_().to(self.device)
		return torch.zeros(self.n_layers, batch_size, self.hid_dim).to(self.device)


class Reconstruction_Model(nn.Module):
	"""
	Reconstruction Model using a GRU-based Encoder-Decoder
	"""

	def __init__(self, in_dim, enc_hid_dim, dec_hid_dim, out_dim, dropout, device='cpu'):
		super(Reconstruction_Model, self).__init__()
		self.dropout = dropout
		self.device = device
		self.enc_hid_dim = enc_hid_dim

		self.encoder = nn.GRU(in_dim, enc_hid_dim, batch_first=True, dropout=dropout)
		self.decoder = nn.GRU(in_dim, dec_hid_dim, batch_first=True, dropout=dropout)
		self.fc = nn.Linear(dec_hid_dim, out_dim)
		self.relu = nn.ReLU()
		# self.hidden = None

	def forward(self, x):
		# x shape: (b, n, k)
		h0 = torch.zeros(1, x.shape[0], self.enc_hid_dim).to(self.device)
		_, enc_last_hid = self.encoder(x, h0)  # last hidden state, will be used as initial hidden state for decoder

		if self.training:
			# Reconstruct in reverse order
			x_flip = torch.flip(x, [1])
			dec_out, _ = self.decoder(x_flip, enc_last_hid)
			x_flip_recon = self.fc(dec_out)

		else:
			x_flip_recon = torch.zeros_like(x)
			for i in range(x.shape[1]):
				dec_out, _ = self.decoder(x_flip_recon, enc_last_hid)
				x_flip_recon[:, i, :] = self.fc(dec_out[:, i, :])

		x_recon = torch.flip(x_flip_recon, [1])
		return x_recon

class Forecasting_Model(nn.Module):
	"""
	Forecasting model (fully-connected network)
	"""
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