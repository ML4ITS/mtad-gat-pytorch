import torch
import torch.nn as nn

from modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel,
    TemporalConvNet,
    VanillaVAE,
    VAE
)


class MTAD_GAT(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2,
        use_tcn = True,
        reduce_dimensionality = True,
        use_vae = True,
        use_KLD = False
    ):
        super(MTAD_GAT, self).__init__()
        self.reduce_dimensionality = reduce_dimensionality

        if self.reduce_dimensionality:
            self.dim_red = nn.Sequential(
                nn.Linear(n_features, 50),
                nn.LeakyReLU()
            )
            self.dim_up = nn.Sequential(
                nn.Linear(50, n_features),
                nn.LeakyReLU()
            )
            n_features = 50
        

        self.use_tcn=use_tcn
        if use_tcn:
            self.tcn1 = TemporalConvNet(n_features, n_features, [n_features, n_features])
            self.tcn2 = TemporalConvNet(1, 1, [10, 10])
            self.tcn3 = TemporalConvNet(n_features, n_features, [n_features, n_features])
        else:
            self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        
        self.use_vae = use_vae
        self.use_KLD = use_KLD
        if self.use_vae:
            self.recon_model = VAE(input_dim=window_size, hidden_dim = 200, latent_dim=gru_hid_dim)
        else:
            self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features

        #
        if self.reduce_dimensionality:
            x = self.dim_red(x)

        if self.use_tcn:
            x = self.tcn1(x)
        else:
            x = self.conv(x)
        
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp

        if self.use_tcn:
            h_end = self.tcn2(h_end)

        predictions = self.forecasting_model(h_end)
        
        if self.use_vae:
            #if using vae the graph convolutional layer is ignored and the input is passed straight to the vae
            recons, mean, log_var = self.recon_model(x.transpose(1,2))
            recons = recons.transpose(1,2)
            if not self.use_KLD:
                mean = 0
                log_var = 0 
        else:
            recons = self.recon_model(h_end)
            mean, log_var = 0
        if self.use_tcn:
            recons = self.tcn3(recons)

        if self.reduce_dimensionality:
          #  predictions = self.dim_up(predictions)
            recons = self.dim_up(recons)

        return predictions, recons, mean, log_var
