import torch.nn as nn
from einops.layers.torch import Rearrange

class connectome_recon(nn.Module):
    def __init__(self, *,
                        connectome_features=4950, #upper triangle edges of connectome
                        dim=128, #taking this from kraken coder
                        emb_dropout=0.5, 
                        decoder_name="LinearDecoder",
                        VAE_flag=False,
                        VAE_latent_dim=128,
                        specific_model=None
                        ):
        super().__init__()

        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = nn.Linear(connectome_features, dim) #upper edges to embedding dim chosen
        
        if decoder_name == "LinearDecoder":
            self.decoder = LinearDecoder(dim=dim, 
                                         connectome_features=connectome_features,
                                         )
        elif decoder_name == "PointwiseDecoder":
            self.decoder = PointwiseDecoder(dim=dim,
                 hidden=512, #iterim large channel number, currently arbitrary might need tuning too
                 connectome_features=connectome_features
                 )
            
    def encode(self, connectome):
        connectome = self.dropout(connectome)
        return self.encoder(connectome)
    
    def decode(self, hidden):
        return self.decoder(hidden)
    
    def forward(self, connectome):
        z = self.encode(connectome)
        connectome_hat_uppertri = self.decode(z) #prediction is vectorized upper triangle
        return connectome_hat_uppertri

######################## DECODER ########################
class LinearDecoder(nn.Module):
    def __init__(self, *, 
                 dim, connectome_features
                 ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, connectome_features),
            nn.BatchNorm1d(connectome_features),
        )

    def forward(self, x):
        #here x should be B, 100, 10
        return self.mlp(x) #ends with B,100,100

class PointwiseDecoder(nn.Module):
    '''
    Made this to experiment with predicting all 15 channels instead of having 15 separate models.
    mimicks linear layer by using 1D convs with kernel=1 but works on each patch (hence kernel=1) separately and
    works on them sequentially. First patch, then next, and so on from dim=192 --> big hidden of 512 then back down to our desired 15.
    '''
    def __init__(self, dim=192, #patch or node embedding dim
                 hidden=512,
                 connectome_features=100
            ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, connectome_features, kernel_size=1),
        )
        # self.norm_last = nn.BatchNorm1d(connectome_features)

    def forward(self, z):
        # return self.net(z)
        z = z.transpose(0,1) #z: Bx1x128 transposed --> Bx128x1
        out = self.net(z)
        # out = self.norm_last(out)
        out = out.transpose(0,1)#.reshape(z.size(0), self.n_patches, self.n_channels, self.n_vertices) #return to original dimes
        return out
        
