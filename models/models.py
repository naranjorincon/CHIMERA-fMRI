import sys
sys.path.append('../') #models
sys.path.append('../../')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import Tensor
from typing import Optional

class SurfaceImageTransformer(nn.Module):
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
                        latent_samples=100
                        ):

        super().__init__()

        patch_dim = num_channels * num_vertices
        mlp_dim = 4*dim

        # inputs has size = b * c * n * v
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        dim_head = math.ceil(dim / heads)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.VAE_flag = VAE_flag
        if VAE_flag is True:
            self.to_latent = nn.Sequential(Rearrange('b n d  -> b (n d)'))
            self.VAE_latent_dim = VAE_latent_dim
            self.latent_samples = latent_samples
            self.fc_mu = nn.Linear(num_patches*dim, VAE_latent_dim)
            self.fc_var = nn.Linear(num_patches*dim, VAE_latent_dim)
    
    def forward(self, img):
        b,c,n,v = img.shape
        mu=0
        log_var=0
        x = self.to_patch_embedding(img)

        x += self.pos_embedding
        x = self.dropout(x)

        z = self.transformer(x)

        # VAE option to think about
        if self.VAE_flag is True:
            z = self.to_latent(z) # collapses into a vector to extract mean and var
            # reparam trick
            mu = self.fc_mu(z)
            log_var = self.fc_var(z)
            std = torch.exp(0.5 * log_var) # make into std

            epsilon = torch.randn(self.latent_samples, b, self.VAE_latent_dim) #torch.randn_like(mu) # think its supposed to be mu
            z_samples = mu.unsqueeze(0) + (std.unsqueeze(0) * epsilon) # reparam trick
            z_average = z_samples.mean(dim=0)
            z = z_average

        return z


############################# HELPER FUNCTIONS #############################
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5 #square root of d_k and it is used as denominator hence **-0.5
        self.norm = nn.LayerNorm(dim) #layer norm instead of batch norm
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, 
                 dim, 
                 depth, 
                 heads, 
                 dim_head, 
                 mlp_dim, 
                 dropout = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class KLAnnealer:
    '''Beta-VAE type. Manipulating KL coefficient.'''
    def __init__(self, model, beta_start=0.0, beta_end=1.0, anneal_steps=2000):
        self.model        = model
        self.beta_start   = beta_start
        self.beta_end     = beta_end
        self.anneal_steps = anneal_steps
        self._step        = 0
        model.beta        = beta_start

    def step(self):
        self._step = min(self._step + 1, self.anneal_steps)
        frac = self._step / self.anneal_steps
        self.model.beta = self.beta_start + frac * (self.beta_end - self.beta_start)

    @property
    def current_beta(self):
        return self.model.beta