import torch
import trimesh
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv
import sys
import numpy as np
from models.models import SurfaceImageTransformer
from einops.layers.torch import Rearrange

class topomap_recon(nn.Module):
    def __init__(self, *, 
                dim=384, 
                depth=6,
                heads=4,
                num_patches=320,
                num_channels=15,
                num_vertices=153,
                # dim_head=64,
                dropout=0.1,
                emb_dropout=0.3,
                VAE_flag=False,
                VAE_latent_dim=100,
                latent_samples=100,
                decoder_name="LinearDecoder",
                specific_model=None
            ):
        
        super().__init__()

        self.num_channels = num_channels
        self.num_patches = num_patches
        self.num_vertices = num_vertices
        # hidden = num_patches*dim if VAE_flag is False else VAE_latent_dim
        # self.hidden = hidden
        self.encoder = SurfaceImageTransformer( 
                    dim=dim, 
                    depth=depth,
                    heads=heads,
                    num_patches=num_patches,
                    num_channels=num_channels,
                    num_vertices=num_vertices,
                    # dim_head=dim_head,
                    dropout=dropout,
                    emb_dropout=emb_dropout,
                    VAE_flag=VAE_flag,
                    VAE_latent_dim=VAE_latent_dim,
                    latent_samples=latent_samples
            )
        # output_dim = num_channels*num_patches*num_verteces
        if num_patches == 80:
            ico_res=1
        elif num_patches == 320:
            ico_res=2
        elif num_patches == 1280:
            ico_res=3

        if decoder_name == "LinearDecoder":
            self.decoder = LinearDecoder(dim=dim, 
                                         n_channels=num_channels,
                                         n_vertices=num_vertices)
        elif decoder_name == "ConvDecoder":
            self.decoder = ConvDecoder(dim=dim, n_channels=num_channels,
                                        n_vertices=num_vertices)
        elif decoder_name == "PointwiseDecoder":
            self.decoder = PointwiseDecoder(dim=dim,
                 hidden=512, #iterim large channel number, currently arbitrary might need tuning too
                 n_channels=num_channels,
                 n_patches=num_patches,
                 n_vertices=num_vertices)
        elif decoder_name == "GATDecoder":
            mesh = trimesh.creation.icosphere(subdivisions=ico_res)
            edge_index = build_edge_index(mesh.face_adjacency)
            self.decoder = GATDecoder(edge_index=edge_index,
                    dim=dim, heads=4,
                    hidden=64, n_channels=num_channels,
                    n_vertices=num_vertices)

    def encode(self, img):
        return self.encoder(img)
    
    def decode(self, hidden):
        return self.decoder(hidden)
    
    def forward(self, img, return_latent=False):
        z = self.encode(img)
        x_hat = self.decode(z)
        if return_latent is True:
            return x_hat, z

        return x_hat

######################## DECODER ########################
class LinearDecoder(nn.Module):
    def __init__(self, *, 
                 dim, n_channels,n_vertices
                 ):
        super().__init__()

        output = n_channels*n_vertices
        self.mlp = nn.Sequential(
            # Rearrange('b n d  -> b (n d)', n=num_patches, d=dim),
            nn.Linear(dim, output), #goes from Bx320xD-->Bx320x(C*V)
            Rearrange('b n (c v)  -> b n c v', c=n_channels, v=n_vertices)
            )

    def forward(self, x):
        out = self.mlp(x)
        return out.permute(0,2,1,3) #final is Bx15x320x153 or BxCxPxV

        return x

class ConvDecoder(nn.Module):
    def __init__(self, *, 
                 dim, n_channels,n_vertices
                 ):
        super().__init__()

        self.rearrange_latent = nn.Sequential(
            nn.Linear(dim, n_channels*n_vertices), #from Bx320x(latent_dim)-->Bx320x(15*153)
            Rearrange('b n (c v)  -> b c n v', c=n_channels, v=n_vertices),
        ) #goes from Bx320*D-->Bx320xD then linear so now Bx320x(15*153) then finally rearrange to Bx15x320x153

        self.mlp = nn.Sequential(
            # nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        ) #conv is just from the latent 15-->true 15

    def forward(self, x):
        x = self.rearrange_latent(x)
        out = self.mlp(x)
        return out #.permute(0,2,1,3)

class PointwiseDecoder(nn.Module):
    '''
    Made this to experiment with predicting all 15 channels instead of having 15 separate models.
    mimicks linear layer by using 1D convs with kernel=1 but works on each patch (hence kernel=1) separately and
    works on them sequentially. First patch, then next, and so on from dim=192 --> big hidden of 512 then back down to our desired 15.
    '''
    def __init__(self, dim=192,
                 hidden=512,
                 n_channels=15,
                 n_patches=320,
                 n_vertices=153
            ):

        super().__init__()

        self.n_patches, self.n_channels, self.n_vertices = n_patches, n_channels, n_vertices
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, n_channels*n_vertices, kernel_size=1)
        )

    def forward(self, z):
        z = z.transpose(1,2) #z: Bx320x192 transposed --> Bx192x320
        out = self.net(z)
        out = out.transpose(1,2).reshape(z.size(0), self.n_patches, self.n_channels, self.n_vertices) #return to original dimes
        return out.permute(0, 2, 1, 3) #finish by returning it to original exact shape

def build_edge_index(face_adjacency):
        row, col = face_adjacency[:, 0], face_adjacency[:, 1]
        row_full = np.concatenate([row, col])
        col_full = np.concatenate([col, row])
        edge_index = torch.tensor(np.stack([row_full, col_full]), dtype=torch.long)
        return edge_index  # shape (2, 960) for 320 faces

def make_batched_edges(edge_index, batch_size, num_nodes):
        """
        edge_index: (2, E) - the single-graph edge index (e.g. from build_edge_index)
        Returns: (2, B*E) - block-diagonal batched edge index, matching node ordering
        of x.reshape(B*num_nodes, D)
        """
        E = edge_index.size(1)
        device = edge_index.device
        offsets = torch.arange(batch_size, device=device).view(batch_size, 1, 1) * num_nodes
        batched = edge_index.unsqueeze(0).expand(batch_size, -1, -1) + offsets # B,2,E
        return batched.permute(1,0,2).reshape(2, batch_size*E)

class GATDecoder(nn.Module):
    def __init__(self, edge_index,
                 dim=192, heads=4,
                 hidden=64, n_channels=15,
                 n_vertices=153):
        super().__init__()

        self.edge_index = edge_index
        # self.register_buffer('edge_index', edge_index) #2xE of triangle/patch adjecency
        self.gat1 = GATv2Conv(dim, hidden, heads=heads)
        self.gat2 = GATv2Conv(hidden*heads, hidden, heads=heads)
        self.head = nn.Linear(hidden*heads, n_channels*n_vertices)
        self.n_vertices = n_vertices
        self.n_channels = n_channels

    def forward(self, z): #z: Bx320x192  --> glatten batch into one big graph, or loop
        '''
        represents nodes as N times B so that all subjects in B and nodes are present. 
        GTA doesn't have "batch" the way other simpler torch modules do.
        '''
        B, N, D = z.shape
        x = z.reshape(B*N, D) 
        batch_edge_index = make_batched_edges(edge_index=self.edge_index,batch_size=B, num_nodes=N)
        h = F.elu(self.gat1(x, batch_edge_index))
        h = F.elu(self.gat2(h, batch_edge_index))
        out = self.head(h).reshape(B,N,self.n_channels, self.n_vertices)
        return out.permute(0,2,1,3)