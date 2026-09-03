# import torch
# import trimesh
import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn.conv import GATv2Conv
# import sys
# import numpy as np
from models.models import BrainGraphTransformer
from einops.layers.torch import Rearrange

# class connectome_recon(nn.Module):
#     '''
#     Brain Graph Transformer taken from "Brain Network Transformer": https://arxiv.org/abs/2210.06681
#     Node features are that node's connectivity profile and it is shown to be good enough to embed graph information, lapalce pos emb doesn't add more
#     or take away. It IS computationlly heavy, so why do it if node conn profile is good enough - so they (the authors) argue. In the paper, each node has its profile (the connectivity) and 
#     vanilla transformer modules are used on those node representations. Conn profile is the "corresponding row of that node". Edge weights are also ignored because computationally expensive and do not
#     seem to make performance better in the specific context of correlation brain ROIs matrices. As such, we use the vanilla transformer module and node feats as their conn profile.
#     '''
#     def __init__(self, *,
#                         parcellation_size_og=100,
#                         dim, # no self loops 
#                         depth, 
#                         heads, 
#                         emb_dropout=0.1, 
#                         dropout=0.3, # drop out used in transformer block
#                         decoder_name="LinearDecoder"
#                         ):
#         super().__init__()
        
#         self.encoder = BrainGraphTransformer( 
#                     enc_model_dim=dim, # no self loops 
#                         depth=depth, 
#                         heads=heads, 
#                         emb_dropout=emb_dropout, 
#                         dropout=dropout
#         )

#         # project to match patch dims
#         if decoder_name == "LinearDecoder":
#             self.decoder = LinearDecoder(dim=dim, 
#                                          parcellation_size_og=parcellation_size_og,
#                                          )
#         elif decoder_name == "ConvDecoder":
#             self.decoder = ConvDecoder(dim=dim, 
#                                        parcellation_size_og=parcellation_size_og)
#         elif decoder_name == "PointwiseDecoder":
#             self.decoder = PointwiseDecoder(dim=dim,
#                  hidden=512, #iterim large channel number, currently arbitrary might need tuning too
#                  parcellation_size_og=parcellation_size_og
#                  )
            
#     def encode(self, connectome):
#         return self.encoder(connectome)
    
#     def decode(self, hidden):
#         return self.decoder(hidden)
    
#     def forward(self, connectome):
#         z = self.encode(connectome)
#         print(z)
#         connectome_hat_uppertri = self.decode(z) #prediction is vectorized upper triangle
#         return connectome_hat_uppertri


class connectome_recon(nn.Module):
    def __init__(self, *,
                        connectome_features=4950, #upper triangle edges of connectome
                        dim=128, #taking this from kraken coder
                        emb_dropout=0.5, 
                        decoder_name="LinearDecoder"
                        ):
        super().__init__()

        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = nn.Linear(connectome_features, dim) #upper edges to embedding dim chosen
        
        if decoder_name == "LinearDecoder":
            self.decoder = LinearDecoder(dim=dim, 
                                         connectome_features=connectome_features,
                                         )
        elif decoder_name == "ConvDecoder":
            self.decoder = ConvDecoder(dim=dim, 
                                       connectome_features=connectome_features)
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
            nn.Linear(dim, connectome_features)
        )

    def forward(self, x):
        #here x should be B, 100, 10
        return self.mlp(x) #ends with B,100,100

class ConvDecoder(nn.Module):
    def __init__(self, *, 
                 dim, connectome_features
                 ):
        super().__init__()

        # #add a single channel
        # self.rearrange_latent = nn.Sequential(
        #     nn.Linear(dim, connectome_features), #from Bx100x(latent_dim)-->Bxparcellation_nx(1*parcellation_n)
        #     # Rearrange('b n (c p)  -> b c n p', c=1, p=connectome_features),
        # )

        self.mlp = nn.Sequential(
            # nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, connectome_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        #here x should be B, 100, 10
        # x = self.rearrange_latent(x)
        return self.mlp(x)

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
            nn.Conv1d(hidden, connectome_features, kernel_size=1)
        )

    def forward(self, z):
        return self.net(z)
        
