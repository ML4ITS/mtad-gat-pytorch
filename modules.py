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
		# For feature attention we represent each node by a sequential vector,
		# containing all its values within the window

		v = x.permute(0, 2, 1)

		#Wh = torch.mm(h, self.W)  # Transformation shared across nodes (not used)

		# Creating matrix of concatenations of node features
		attn_input = self._make_attention_input(v)  # (b, k, k, 2*win_size)

		# Attention scores
		e = self.leakyrelu(torch.matmul(attn_input, self.w)).squeeze(3)	# (b, k, k, 1)

		# Attention weights
		attention = torch.softmax(e, dim=2)
		attention = torch.dropout(attention, self.dropout, train=self.training)

		# Computing new node features using the attention
		h = torch.matmul(attention, v)

		return h

	def _make_attention_input(self, v):
		""" Preparing the feature attention mechanism.
			Creating matrix with all possible combinations of concatenations of node.
			Each node consists of all values of that node within the window
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

		# Creating matrix of concatenations of node features
		attn_input = self._make_attention_input(x)

		# Attention scores
		e = self.leakyrelu(torch.matmul(attn_input, self.w).squeeze(3))

		# Attention weights
		attention = torch.softmax(e, dim=2)
		attention = torch.dropout(attention, self.dropout, train=self.training)

		# Computing new node features using the attention
		h = torch.matmul(attention, x)

		return h



	def _make_attention_input(self, v):
		""" Preparing the temporal attention mechanism.
			Creating matrix with all possible combinations of concatenations of node values:
				(v1, v2..)_t1 || (v1, v2..)_t1
				(v1, v2..)_t1 || (v1, v2..)_t2

				...
				...

				(v1, v2..)_tn || (v1, v2..)_t1
				(v1, v2..)_tn || (v1, v2..)_t2

		"""

		# v has shape (b, n, k)

		K = self.window_size
		Wh_blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
		Wh_blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix

		combined = torch.cat((Wh_blocks_repeating, Wh_blocks_alternating), dim=2)  # Shape (b, K*K, 2*window_size)
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
		# self.hidden = None

	def forward(self, x):
		h0 = torch.zeros(self.n_layers, x.shape[0], self.hid_dim).to(self.device)
		out, h = self.gru(x, h0)
		return out, h


class RNNEncoder(nn.Module):
	"""
	    Encoder network containing enrolled GRU
	    :param in_dim: number of input features
	    :param hid_dim: hidden size of the RNN
	    :param n_layers: number of layers in RNN
	    :param dropout: percentage of nodes to dropout
	    :param device: which device to use (cpu or cuda)
	"""

	def __init__(self, in_dim, n_layers=1, hid_dim=64, dropout=0.0, device='cpu'):
		super(RNNEncoder, self).__init__()

		self.in_dim = in_dim
		self.n_layers = n_layers
		self.hid_dim = hid_dim

		self.rnn = nn.GRU(input_size=in_dim, hidden_size=hid_dim, num_layers=1, batch_first=True, dropout=dropout)
		#self.rnn2 = nn.GRU(self.hid_dim, self.embed_dim, n_layers, batch_first=True, dropout=dropout)

	def forward(self, x):
		"""Forward propagation of encoder. Given input, outputs the last hidden state of encoder
			:param x: input to the encoder, of shape (batch_size, window_size, number_of_features)
			:return: last hidden state of encoder, of shape (batch_size, hidden_size)
		"""

		_, h_end = self.rnn(x)
		return h_end

class RNNDecoder(nn.Module):
	"""
		    Decoder network converts latent vector into output
		    :param window_size: length of the input sequence
		    :param batch_size: batch size of the input sequence
		    :param in_dim: number of input features
		    :param n_layers: number of layers in RNN
		    :param hid_dim: hidden size of the RNN
		    :param batch_size: batch size
		    :param dropout: percentage of nodes to dropout
		    :param device: which device to use (cpu or cuda)
		"""
	def __init__(self, in_dim, n_layers, hid_dim, out_dim, dropout=0.0, device='cpu'):
		super(RNNDecoder, self).__init__()

		self.in_dim = in_dim

		self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)
		#self.decoder_inputs = torch.zeros(self.window_size, batch_size, 1, requires_grad=True).to(device)

		self.fc = nn.Linear(hid_dim, out_dim)

	def forward(self, h):

		decoder_out, _ = self.rnn(h)
		# print(f'decoder_out: {decoder_out.shape}')

		out = self.fc(decoder_out)
		return out

class RNNAutoencoder(nn.Module):
	"""
	Reconstruction Model using a GRU-based Encoder-Decoder.
	TODO
	"""

	def __init__(self,
				 window_size,
				 n_features,
				 n_layers,
				 hid_dim,
				 out_dim,
				 dropout,
				 device='cpu'):
		super(RNNAutoencoder, self).__init__()

		self.window_size = window_size
		self.dropout = dropout
		self.device = device

		self.encoder = RNNEncoder(3*n_features, n_layers, hid_dim, dropout=dropout, device=device)
		self.decoder = RNNDecoder(hid_dim, n_layers, hid_dim, out_dim, dropout=dropout, device=device)

	def forward(self, x):
		# x shape: (b, n, k)

		h_end = self.encoder(x).squeeze(0).unsqueeze(2)

		# print(f'h_end: {h_end.shape}')
		h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
		# print(h_end_rep.shape)

		out = self.decoder(h_end_rep)
		# print(f'out: {out.shape}')
		return out

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