"""
Fusion
------
Two strategies for getting a shared z from SiT patch tokens:

  Strategy A — Pool first, fuse flat:
      SiT z  (B, 320, 192)  → mean-pool patches → (B, 192) useful becasue patches*dim linear layer to output woudl be huge.
      Conn z (B, latent)    → as-is
      PoE/MoE on (B, D) vectors → z_shared (B, D)
      Decode z_shared → both modalities

  Strategy B — Fuse patch-wise, keep spatial structure:
      SiT z  (B, 320, 192)  → each patch gets its own (mu, lv)
      Conn z (B, latent)    → broadcast to (B, 320, latent)
      PoE/MoE per patch     → z_shared (B, 320, D)
      Decode z_shared → both modalities (surface decoder sees full patch structure)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# ── numerical safety/clamping of grads ───────────────────────────────────────\
LOG_VAR_MAX =  200.0
LOG_VAR_MIN = -1 * LOG_VAR_MAX
EPS         =  1e-99

# use pre-trained encoders. Currently just regular deterministic z, but will be adapted to ue VAEs later
class FrozenSiTExpert(nn.Module):
    """
    Wraps a pretrained SiT encoder and adds trainable mu/log_var heads if needed

    The SiT encoder is fully frozen. The two small linear heads are the
    only trainable parameters in this module. They learn to read a
    probabilistic distribution out of the frozen patch embeddings.

    Args:
        sit_encoder  : pre-trained nn.Module of SiT
        sit_latent   : SiT specific altent dim  (192-tiny)
        fusion_dim   : shared latent across all encoders
        pool_patches : if True  → pool (B,320,192) → (B,192) before heads
                       if False → keep patch structure, heads run patch-wise
                                  output is (B,320,fusion_dim)
    """

    def __init__(
        self,
        sit_encoder: nn.Module,
        sit_latent: int  = 192,
        fusion_dim: int  = 128,
        pool_patches: bool = True,
        ):
        super().__init__()
        self.pool_patches = pool_patches

        # freeze 
        self.encoder = sit_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # add VAE component here, learnable and separate from encoder part. TODO is make the encoder 
        # VAE itself and compare? Althought same order of things so maybe not necessary.
        self.fc_mu      = nn.Linear(sit_latent, fusion_dim) #TODO is make these convolutions??
        self.fc_log_var = nn.Linear(sit_latent, fusion_dim)

        # Initialise log_var head to zero so initial posterior comes from gaussian N(0,I)
        nn.init.zeros_(self.fc_log_var.weight)
        nn.init.zeros_(self.fc_log_var.bias)
    
    def encoder_only(self, x: Tensor):
        return self.encoder.encode(x) 
        
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        x : (B, 15, 320, 153)
        Returns mu, log_var each shaped:
            (B, fusion_dim)       if pool_patches=True
            (B, 320, fusion_dim)  if pool_patches=False
        """
        with torch.no_grad():
            z = self.encoder_only(x) # (B, 320, 192) — frozen

        if self.pool_patches:
            z = z.mean(dim=1)            # (B, 192) — mean over patches

        mu      = self.fc_mu(z)
        log_var = self.fc_log_var(z).clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        return mu, log_var


class FrozenConnectomeExpert(nn.Module):
    """
    Wraps a pretrained connectome encoder and adds trainable mu/log_var heads. Same as we do with SiT above.

    Args:
        conn_encoder : connectome nn.Module
        conn_latent  : latent dim for connectome should be 256
        fusion_dim   : target latent dim for the shared space
    """

    def __init__(
        self,
        conn_encoder: nn.Module,
        conn_latent: int=256,
        fusion_dim: int=128,
        ):
        super().__init__()

        # ── freeze ────────────────────────────────────────────────────────────
        self.encoder = conn_encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.fc_mu      = nn.Linear(conn_latent, fusion_dim)
        self.fc_log_var = nn.Linear(conn_latent, fusion_dim)

        #init as zeros for normal gaussian init as with SiT
        nn.init.zeros_(self.fc_log_var.weight)
        nn.init.zeros_(self.fc_log_var.bias)
    
    def encoder_only(self, x: Tensor):
        return self.encoder.encode(x)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        x : (B, N_tri)
        Returns mu, log_var each shaped (B, fusion_dim)
        """
        with torch.no_grad():
            # z = self.encoder(x)          # (B, conn_latent) — frozen
            z = self.encoder_only(x)

        mu      = self.fc_mu(z)
        log_var = self.fc_log_var(z).clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        return mu, log_var


# ═════════════════════════════════════════════════════════════════════════════
# Decoders
# ═════════════════════════════════════════════════════════════════════════════

class SurfaceDecoder(nn.Module):
    """
    Decodes a shared latent z back to surface shape (B, C, P, V).

    Strategy A (pool_patches=True, z is flat):
        z : (B, fusion_dim)
        → expand to (B, 320, fusion_dim)  [same vector broadcast to each patch]
        → linear fusion_dim → 15*153
        → reshape to (B, 15, 320, 153)

    Strategy B (pool_patches=False, z keeps patch structure):
        z : (B, 320, fusion_dim)
        → linear fusion_dim → 15*153
        → reshape to (B, 15, 320, 153)
    """

    def __init__(
        self,
        fusion_dim: int  = 128,
        n_patches: int   = 320,
        n_networks: int  = 15,
        n_vertices: int  = 153,
        pool_patches: bool = True,
    ):
        super().__init__()
        self.n_patches   = n_patches
        self.n_networks  = n_networks
        self.n_vertices  = n_vertices
        self.pool_patches = pool_patches

        self.proj = nn.Linear(fusion_dim, n_networks * n_vertices)

    def forward(self, z: Tensor) -> Tensor:
        """
        z : (B, fusion_dim)         if pool_patches=True
            (B, 320, fusion_dim)    if pool_patches=False
        Returns (B, 15, 320, 153)
        """
        B = z.shape[0]

        if self.pool_patches:
            # Broadcast the same flat vector to every patch
            z = z.unsqueeze(1).expand(B, self.n_patches, -1)   # (B, 320, fusion_dim)

        # z is now (B, 320, fusion_dim) in both strategies
        out = self.proj(z)                                       # (B, 320, 15*153)
        out = out.view(B, self.n_patches, self.n_networks, self.n_vertices)
        out = out.permute(0, 2, 1, 3)                           # (B, 15, 320, 153)
        return out


class ConnectomeDecoder(nn.Module):
    """
    Decodes a shared latent z back to a connectome upper triangle (B, N_tri).
    Simple MLP with one hidden layer.
    """

    def __init__(self, fusion_dim: int, n_tri: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_tri),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        z : (B, fusion_dim)         flat vector 
        Returns (B, N_tri)
        """
        # if z.dim() == 3:
        #     # If z still has patch dim (Strategy B), pool before connectome decode
        #     z = z.mean(dim=1)
        return self.net(z)


# ═════════════════════════════════════════════════════════════════════════════
# PoE and MoE fusion math
# ═════════════════════════════════════════════════════════════════════════════

def product_of_experts(
    mu_list: list[Tensor],
    log_var_list: list[Tensor],
) -> tuple[Tensor, Tensor]:
    """
    Product of Experts fusion. Works on any shape — flat (B,D) or
    patch-structured (B,P,D) — as long as all inputs share the same shape.

    Always prepends a unit Gaussian prior expert N(0,I) so the product
    is well-defined even with a single modality.

    Returns mu_joint, log_var_joint — same shape as inputs.
    """
    device = mu_list[0].device
    prior_mu  = torch.zeros_like(mu_list[0])
    prior_lv  = torch.zeros_like(log_var_list[0])

    mus  = torch.stack([prior_mu]  + mu_list,      dim=0)   # (E+1, ...)
    lvs  = torch.stack([prior_lv]  + log_var_list, dim=0)
    lvs  = lvs.clamp(LOG_VAR_MIN, LOG_VAR_MAX)

    precision = torch.exp(-lvs)                              # (E+1, ...)
    prec_sum  = precision.sum(dim=0)                         # (...)
    mu_joint  = (mus * precision).sum(dim=0) / (prec_sum + EPS)
    lv_joint  = -torch.log(prec_sum + EPS).clamp(LOG_VAR_MIN, LOG_VAR_MAX)

    return mu_joint, lv_joint


def mixture_of_experts(
    mu_list: list[Tensor],
    log_var_list: list[Tensor],
) -> tuple[Tensor, Tensor]:
    """
    Mixture of Experts fusion — approximated as the mean of per-expert
    parameters (MoE mean and variance via moment matching).

    MoE mean     : mu_moe    = (1/M) * sum_m mu_m
    MoE variance : var_moe   = (1/M) * sum_m (var_m + mu_m^2) - mu_moe^2
                             [law of total variance]

    This gives a single Gaussian approximation to the mixture, which is
    cheaper than sampling from each expert separately but captures the
    higher variance of the mixture relative to any single expert.

    Returns mu_moe, log_var_moe — same shape as inputs.
    """
    mus  = torch.stack(mu_list,      dim=0)                  # (M, ...)
    lvs  = torch.stack(log_var_list, dim=0).clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    vars_ = torch.exp(lvs)                                   # (M, ...)

    mu_moe  = mus.mean(dim=0)                                # (...)
    # E[X^2] - (E[X])^2  where E[X^2] = mean(var + mu^2)
    var_moe = (vars_ + mus.pow(2)).mean(dim=0) - mu_moe.pow(2)
    var_moe = var_moe.clamp(min=EPS)
    lv_moe  = torch.log(var_moe).clamp(LOG_VAR_MIN, LOG_VAR_MAX)

    return mu_moe, lv_moe

# def reparameterise(mu: Tensor, log_var: Tensor) -> Tensor:
#     if not torch.is_grad_enabled():
#         return mu
#     std = torch.exp(0.5 * log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX))
#     return mu + torch.randn_like(std) * std

def reparameterise(mu: Tensor, log_var: Tensor, model_type: str="MoE", latent_samples: int=1000) -> Tensor:
    """
    z = mu + eps * std,  eps ~ N(0, I) but sampled many times then averaged
    Returns mu directly during eval (no stochasticity).
    """
    if not torch.is_grad_enabled():          # inference / eval
        return mu

    b, d = mu.shape #should be 2D of batch x latent_dim_z
    log_var = log_var.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    std = torch.exp(0.5 * log_var)
    if model_type == "MoE":
        eps = torch.randn_like(std)
        return mu + eps * std
    elif model_type == "PoE": # multiple samples when using PoE
        epsilon = torch.randn(latent_samples, b, d) #torch.randn_like(std) #i think its supposed to be mu
        z_samples = mu.unsqueeze(0) + (std.unsqueeze(0) * epsilon) # reparam trick
        z_average = z_samples.mean(dim=0)
        return z_average