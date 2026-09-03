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
        # self.to_latent = nn.Sequential(Rearrange('b n d  -> b (n d)'))
        # self.VAE_flag = VAE_flag
        # if VAE_flag is True:
        #     self.VAE_latent_dim = VAE_latent_dim
        #     self.latent_samples = latent_samples
        #     self.fc_mu = nn.Linear(num_patches*dim, self.VAE_latent_dim)
        #     self.fc_var = nn.Linear(num_patches*dim, self.VAE_latent_dim)
    
    def forward(self, img):
        b,c,n,v = img.shape
        x = self.to_patch_embedding(img)

        x += self.pos_embedding
        x = self.dropout(x)

        z = self.transformer(x)
        # z = self.to_latent(x) # collapses into a vector to extract mean and var

        #VAE option to think about
        # if self.VAE_flag is True:
        #     # reparam trick
        #     mu = self.fc_mu(z)
        #     log_var = self.fc_var(z)
        #     std = torch.exp(0.5 * log_var) # make into std

        #     epsilon = torch.randn(self.latent_samples, b, self.VAE_latent_dim) #torch.randn_like(mu) # think its supposed to be mu
        #     z_samples = mu.unsqueeze(0) + (std.unsqueeze(0) * epsilon) # reparam trick
        #     z_average = z_samples.mean(dim=0)
        #     z = z_average

        return z

# class BrainGraphTransformer(nn.Module):
#     '''
#     Brain Graph Transformer taken from "Brain Network Transformer": https://arxiv.org/abs/2210.06681
#     Node features are that node's connectivity profile and it is shown to be good enough to embed graph information, laplace pos emb doesn't add more
#     or take away. It IS computationlly heavy, so why do it if node conn profile is good enough - so they (the authors) argue. In the paper, each node has its profile (the connectivity) and 
#     vanilla transformer modules are used on those node representations. Conn profile is the "corresponding row of that node". Edge weights are also ignored because computationally expensive and do not
#     seem to make performance better in the specific context of correlation brain ROIs matrices. As such, we use the vanilla transformer module and node feats as their conn profile.

#     Input is the connectome itself, as is. 100x100 or 360x360 and so on.
#     '''
#     def __init__(self, *,
#                         enc_model_dim, # no self loops 
#                         depth, 
#                         heads, 
#                         emb_dropout=0.1, 
#                         dropout=0.3
#                         ):
#         super().__init__()

#         transformer_FFN_dim= 4*enc_model_dim #4*d_model according to DeiT feedforward in transformer architecture
#         self.dropout = nn.Dropout(emb_dropout) # embedding drop out
#         enc_dim_head = math.ceil(enc_model_dim / heads) # based on DeiT should be ceil(model_dim/heads)
#         self.transformer = Transformer(enc_model_dim, depth, heads, enc_dim_head, transformer_FFN_dim, dropout)
        
#     def forward(self, connectome):
#         connectome = self.dropout(connectome)
#         # Pass to Transformer
#         connectome = self.transformer(connectome)
#         return connectome
        

# def vector2mat(self, x:Tensor) -> Tensor:
#     """
#     connectome reconstruction/translations are giving us just the vectorized upper triangle. So making it into a mat here.
#     This is just a tranformation, so not part of the re-weighting/learning process.
#     """
#     b = x.shape[0]
#     device=x.device
#     mat = torch.zeros(b, self.N, self.N, device=device) #make 3D tensor of shape batch,N,N
#     #tri indices
#     offset=-1 #make diagonal 1
#     row_idx, col_idx = torch.tril_indices(
#         self.N, self.N, offset=offset, device=device)
#     #fill lower tri
#     mat[:, row_idx, col_idx]=x
#     mat = mat + mat.transpose(-1,-2) #transpose alst one and second to last one casue 2D matrix inside 3D tensor?
#     mat.diagonal(dim1=-2, dim2=-1).fill_(1.0)
#     return mat

    
def PoE(
        mu_list: list[Tensor],
        log_var_list: list[Tensor],
        eps: float = 1e-8,
        LOG_VAR_MIN:float=-100,
        LOG_VAR_MAX: float=100) -> tuple[Tensor, Tensor]:
    '''
    Given a list of mu_i and logvar_i, compute the product of these gaussians.
    The PoE of N Gaussians is itself a Gaussian:
        var_joint^{-1} = sum_i(var_i^{-1})
        mu_joint       = var_joint * sum_i(mu_i * var_i^{-1})

    Args:
        mu_list:      list of (batch, latent_dim) tensors
        log_var_list: list of (batch, latent_dim) tensors
    Returns:
        mu_joint, log_var_joint & both will be (batch, latent_dim)
    '''
    # Prepend the prior expert: N(0, I) for regularization and stability
    batch, latent_dim = mu_list[0].shape
    device = mu_list[0].device
    prior_mu      = torch.zeros(batch, latent_dim, device=device) #init as such with zeros
    prior_log_var = torch.zeros(batch, latent_dim, device=device)
    # introducing encodings
    all_mu      = [prior_mu]      + mu_list
    all_log_var = [prior_log_var] + log_var_list
    # Stack to (num_experts, batch, latent_dim) so for now should be (2,32,100) and 2==SiT+BNT
    mu_stack      = torch.stack(all_mu,      dim=0)
    log_var_stack = torch.stack(all_log_var, dim=0)
    # clamping due to NaNs earlier 
    # self.LOG_VAR_MIN, self.LOG_VAR_MAX = LOG_VAR_MIN, LOG_VAR_MAX
    log_var_stack = log_var_stack.clamp(LOG_VAR_MIN,LOG_VAR_MAX)
    # Precision = 1 / variance = exp(-log_var)
    precision_stack = torch.exp(-log_var_stack) + eps          # (E, B, D)
    # Joint precision and mean
    precision_joint = precision_stack.sum(dim=0)               # (B, D)
    mu_joint        = (mu_stack * precision_stack).sum(dim=0) / precision_joint + eps
    log_var_joint   = -torch.log(precision_joint + eps)
    log_var_joint   = log_var_joint.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
 
    return mu_joint, log_var_joint

def reparameterise(mu: Tensor, log_var: Tensor, LOG_VAR_MIN: float, LOG_VAR_MAX: float) -> Tensor:
    """
    z = mu + eps * std,  eps ~ N(0, I)
    Returns mu directly during eval (no stochasticity).
    """
    if not torch.is_grad_enabled():          # inference / eval
        return mu
    #also help stabalize log_var herein, not just in PoE
    # LOG_VAR_MIN, LOG_VAR_MAX = self.LOG_VAR_MIN, self.LOG_VAR_MAX
    log_var = log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

    # multiple samples
    # epsilon = torch.randn(self.latent_samples, b, self.VAE_latent_dim) #torch.randn_like(std) #i think its supposed to be mu
    # z_samples = mu.unsqueeze(0) + (std.unsqueeze(0) * epsilon) # reparam trick
    # z_average = z_samples.mean(dim=0)
    # latent_step_identity = z_average #should be shape (batch, latent_dim)

def clip_gradients(model: nn.Module, max_norm:float = 10.0) -> float:
    '''
    noticed some instability with MVAE that happened at MSE and KL so must be stabalized
    for gradients and log_var but also for KL divergence later
    '''
    return nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()

class MVAE(nn.Module):
    '''Multimodal VAE using PoE
    
    Args:
        encoders:   list of ModalityEncoder — one per modality
        decoders:   list of ModalityDecoder — one per modality (same order)
        latent_dim: dimensionality of the shared latent space
        beta:       weight on the KL term (beta-VAE style; default 1.0)
 
    Forward:
        inputs: list[Optional[Tensor]] — one per modality, or None if missing
        Returns an MVAEOutput dataclass.
 
    Notes on missing modalities:
        Pass None in the inputs list for any modality that is absent.
        Only present modalities contribute experts to the PoE product.
        The prior expert N(0,I) always ensures the product is valid.
        
    '''
    def __init__(
        self,
        encoders: list,
        decoders: list,
        latent_dim: int,
        beta: float=1.0, #beta for VAE KL section so beta VAE
        log_var_min: float=-100.0,
        log_var_max: float=100.0
    ):
        
        super().__init__()
        assert len(encoders) == len(decoders), f"Each encoder must have a decoder and vice versa. encs({len(encoders)}) decs({len(decoders)})"

        self.encoders     = nn.ModuleList(encoders)
        self.decoders     = nn.ModuleList(decoders)
        self.latent_dim   = latent_dim
        self.beta         = beta
        self.n_modalities = len(encoders)
        self.LOG_VAR_MIN, self.LOG_VAR_MAX = log_var_min, log_var_max

    def encode(
            self,
            inputs: list[Optional[Tensor]]
    ) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        '''
        Returns:
        mu_joint:        (batch, latent_dim)
        log_var_joint:   (batch, latent_dim)
        modality_params: list of (mu_i, log_var_i) for each present modality
        '''

        mu_list, log_var_list, modality_params = [],[],[] #init empty lists

        # for each encoder and its respective data object/type, get their respective mu_i,log_var_i
        for i, (encoder, x) in enumerate(zip(self.encoders, inputs)):
            if x is None: #if encoder is missing data, inits it with None so that it works even with missing data.
                modality_params.append(None)
                continue
            
            mu_i, log_var_i = encoder(x)
            #clip immediately after encoding
            log_var_i = log_var_i.clamp(self.LOG_VAR_MIN, self.LOG_VAR_MAX)
            mu_list.append(mu_i)
            log_var_list.append(log_var_i)
            modality_params.append((mu_i, log_var_i))
        
        if len(mu_list) == 0:
            raise ValueError("No modalities passed. Neets at least one.")
        
        mu_joint, log_var_joint = PoE(mu_list, log_var_list, self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        return mu_joint, log_var_joint, modality_params
        
    def decode(self, z:Tensor) -> list[Tensor]:
        '''Decode each z with corresponding decoders'''
        return [decoder(z) for decoder in self.decoders]
    
    def compute_kl(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Analytical KL divergence: KL[ q(z|x) || p(z) ]
        where p(z) = N(0, I).

        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        Returns a scalar (mean over batch and latent dims).
        """
        # kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        # return kl.sum(dim=-1).mean()
        LOG_VAR_MIN, LOG_VAR_MAX = self.LOG_VAR_MIN, self.LOG_VAR_MAX
        KL_MAX_PER_DIM=100
        log_var = log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        
        # kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        # kl_mean = kl.mean(dim=0)
        # return kl_mean    
        
        # Hard cap per element to prevent a single dimension blowing up the loss
        kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
        kl_per_dim = kl_per_dim.clamp(max=KL_MAX_PER_DIM)
        return kl_per_dim.sum(dim=-1).mean() 
    
    def elbo(
        self,
        inputs: list[Optional[Tensor]],
        # recon_loss_fn,
        mu: Tensor,
        log_var: Tensor,
        reconstructions: list[Tensor],
        recon_weights: Optional[list[float]] = None # suface maps and connectomes have different ranges of values so trunig to pass these seperately
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute the ELBO = reconstruction_loss + beta * KL. beta will be annealed

        Args:
            inputs:          original inputs (None for missing modalities)
            recon_loss_fn:   callable(recon, target) → scalar loss per modality
                            e.g. nn.MSELoss() or nn.BCEWithLogitsLoss()
            mu, log_var:     joint posterior parameters
            reconstructions: list of decoded tensors

        Returns:
            total_loss, recon_loss, kl_loss  — all scalars
        """

        if recon_weights is None:
            recon_weights = [1.0]*self.n_modalities # num of encoders [1.0 1.0] for SiT and BNT for exmaple

        recon_loss = torch.tensor(0.0, device=mu.device)
        n_present  = 0

        for x, recon, w in zip(inputs, reconstructions, recon_weights):
            if x is None:
                continue
                # recon_loss = recon_loss + recon_loss_fn(recon, x)
                # n_present += 1

            mse = ((x - recon)**2).mean() #MSE with torch values across batch and features
            recon_loss = recon_loss + w*mse # weights encoders differenly so trying to make it more fair for connectomes
            n_present += 1 #num of encoders if any missing will ignore

        recon_loss = recon_loss / max(n_present, 1)   # average over present modalities
        kl_loss    = self.compute_kl(mu, log_var)
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(
        self,
        inputs: list[Optional[Tensor]]
    ) -> dict:
        """
        Full forward pass.

        Args:
            inputs: list of length n_modalities; pass None for missing modalities.

        Returns dict with keys:
            z              — sampled latent (batch, latent_dim)
            mu             — joint posterior mean
            log_var        — joint posterior log-variance
            reconstructions — list of decoded tensors (one per modality)
            modality_params — per-modality (mu_i, log_var_i) or None if missing
        """
        mu, log_var, modality_params = self.encode(inputs)
        z = reparameterise(mu, log_var, self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        reconstructions = self.decode(z)

        # kl_div = self.compute_kl(mu, log_var)
        # kl_loss = self.beta * kl_div

        return {
            "z":               z,
            "mu":              mu,
            "log_var":         log_var,
            "reconstructions": reconstructions,
            "modality_params": modality_params,
            # "kl_loss": kl_loss,
        }
    
        @torch.no_grad() # when calling no_grad yes? so at evaluation step/test step!!
        def sample(self, n_samples:int, device: torch.device) -> list[Tensor]:
            """Sample from standard gaussian space converged to, ideally. N(0,I) for decoding."""
            z = torch.randn(n_samples, self.latent_dim, device=device)
            return self.decode(z)

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