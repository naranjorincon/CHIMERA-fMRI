"""
poe_model.py
============
Product-of-Experts multimodal fusion for SiT surface + connectome data.

Supports:
  - Strategy A: pool SiT patches to flat vector before fusion
  - Strategy B: keep patch structure, fuse patch-wise

All translation directions at inference:
  surface  → surface  (autoencoding baseline)
  connectome → connectome (autoencoding baseline)
  surface  → connectome (cross-modal)
  connectome → surface  (cross-modal)
  surface + connectome → both (joint, best quality)
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import warnings
import torch.nn.functional as F

from models.experts import (
    FrozenSiTExpert,
    FrozenConnectomeExpert,
    SurfaceDecoder,
    ConnectomeDecoder,
    product_of_experts,
    reparameterise,
    LOG_VAR_MIN, LOG_VAR_MAX, EPS,
)


class PoEMultimodalModel(nn.Module):
    """
    Product-of-Experts fusion of SiT surface embeddings and connectome
    embeddings into a shared latent space.

    Args:
        sit_encoder    : pretrained SiT nn.Module
        conn_encoders  : dict mapping ROI label → pretrained connectome encoder
                         e.g. {"100": enc100, "200": enc200, ...}
        conn_latents   : dict mapping ROI label → encoder output dim
                         e.g. {"100": 256, "200": 512, ...}
        conn_n_tri     : dict mapping ROI label → upper triangle length
                         e.g. {"100": 4950, "200": 19900, ...}
        fusion_dim     : shared latent dimensionality (recommend 128–256)
        sit_latent     : SiT encoder output dim per patch (192)
        pool_patches   : True  = Strategy A (flat fusion)
                         False = Strategy B (patch-wise fusion)
        beta           : KL weight; start at 0 and anneal up
        free_bits      : min KL per dim (prevents posterior collapse)
        conn_hidden    : hidden dim in connectome decoder MLP
    """

    def __init__(
        self,
        sit_encoders:  dict[str, nn.Module],
        topo_networks: dict[str, int],
        conn_encoders: dict[str, nn.Module],
        conn_latents:  dict[str, int],
        conn_n_tri:    dict[str, int],
        fusion_dim:    int = 128,
        sit_latent:    int = 192,
        pool_patches:  bool = True,
        beta:          float = 0.0,
        free_bits:     float = 0.1,
        conn_hidden:   int = 512,
        ):
        super().__init__()
        self.fusion_dim   = fusion_dim
        self.pool_patches = pool_patches
        self.beta         = beta
        self.free_bits    = free_bits
        self.conn_keys    = list(conn_encoders.keys())
        self.topo_keys    = list(sit_encoders.keys())

        # ── frozen experts with trainable mu/lv heads ─────────────────────────
        self.surface_experts = nn.ModuleDict({
            k: FrozenSiTExpert(
                sit_encoders[k],
                sit_latent=sit_latent,
                fusion_dim=fusion_dim,
                pool_patches=pool_patches,
            )
            for k in self.topo_keys
        })

        self.conn_experts = nn.ModuleDict({
            k: FrozenConnectomeExpert(
                conn_encoders[k],
                conn_latent=conn_latents[k],
                fusion_dim=fusion_dim,
            )
            for k in self.conn_keys
        })

        # ── decoders (all trainable from scratch) ─────────────────────────────
        self.surface_decoder = nn.ModuleDict({
            k: SurfaceDecoder(
                fusion_dim=fusion_dim,
                n_networks=topo_networks[k],
                pool_patches=pool_patches,
            )
            for k in self.topo_keys
        })

        self.conn_decoders = nn.ModuleDict({
            k: ConnectomeDecoder(
                fusion_dim=fusion_dim,
                n_tri=conn_n_tri[k],
                hidden_dim=conn_hidden,
            )
            for k in self.conn_keys
        })

    # ── encode each present modality ──────────────────────────────────────────
    def encode_surface(self, x: Optional[Tensor], key: str):
        """Returns (mu, log_var) or None."""
        if x is None:
            return None
        return self.surface_experts[key](x)

    def encode_connectome(self, c: Optional[Tensor], key: str):
        """Returns (mu, log_var) or None."""
        if c is None:
            return None
        return self.conn_experts[key](c)

    # ── PoE fusion ────────────────────────────────────────────────────────────

    def fuse(
        self,
        topo_params:    dict[str, Optional[tuple]], #Optional[tuple],
        conn_params:    dict[str, Optional[tuple]],
    ) -> tuple[Tensor, Tensor]:
        """
        Fuse all present modality posteriors via PoE.

        surface_params : (mu, lv) or None
        conn_params    : {"100": (mu, lv) or None, "200": ..., ...}

        Returns z_shared (sampled), mu_joint, lv_joint.
        """
        mu_list, lv_list = [], []

        for k, params in topo_params.items():
            if params is not None:
                mu_list.append(params[0])
                lv_list.append(params[1])

        for k, params in conn_params.items():
            if params is not None:
                mu_list.append(params[0])
                lv_list.append(params[1])

        if len(mu_list) == 0:
            raise ValueError("At least one modality must be provided.")

        mu_j, lv_j = product_of_experts(mu_list, lv_list)
        z = reparameterise(mu_j, lv_j, model_type="PoE") #for PoE we do sample then average
        return z, mu_j, lv_j

    # ── decode ────────────────────────────────────────────────────────────────

    def decode_surface(self, z: Tensor, key: str) -> Tensor:
        return self.surface_decoder[key](z)

    def decode_connectome(self, z: Tensor, key: str) -> Tensor:
        return self.conn_decoders[key](z)

    def decode_all(self, z: Tensor) -> dict:
        """Decode z into all modalities."""
        # out = {"surface": self.decode_surface(z)}
        out={}
        for k in self.topo_keys:
            out[f"topo_{k}"] = self.decode_surface(z, k)

        for k in self.conn_keys:
            out[f"conn_{k}"] = self.decode_connectome(z, k)
        return out

    # ── loss ──────────────────────────────────────────────────────────────────

    def compute_kl(self, mu: Tensor, lv: Tensor) -> Tensor:
        lv_s = lv.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        kl = -0.5 * (1 + lv_s - mu.pow(2) - lv_s.exp())
        if self.free_bits > 0:
            kl = kl.clamp(min=self.free_bits)
        return kl.clamp(max=200.0).sum(dim=-1).mean()

    def compute_loss(
        self,
        topomaps:     dict[str, Optional[Tensor]],
        connectomes:  dict[str, Optional[Tensor]],
        recon_weights: Optional[dict] = None,
    ) -> dict:
        """
        Full forward pass + loss computation.

        Args:
            surface     : (B, 15, 320, 153) or None
            connectomes : {"100": (B, 4950) or None, ...}
            recon_weights: optional per-modality MSE weights
                           e.g. {"surface": 1.0, "conn_100": 0.1}
                           Use if modalities have very different scales.

        Returns dict with keys:
            loss, recon_loss, kl_loss,
            z, mu, lv,
            reconstructions (dict of all decoded outputs)
        """
        if recon_weights is None:
            recon_weights = {}

        # Encode
        topo_params = {
            k: self.encode_surface(topomaps.get(k), k)
            for k in self.topo_keys
        }
        conn_params = {
            k: self.encode_connectome(connectomes.get(k), k)
            for k in self.conn_keys
        }

        # Fuse via PoE
        z, mu, lv = self.fuse(topo_params, conn_params)

        # Decode all
        recons = self.decode_all(z)

        # Reconstruction loss — only on present modalities
        recon_loss = torch.tensor(0.0, device=z.device)
        n_present  = 0
        for k in self.topo_keys:
            x = topomaps.get(k)
            if x is not None:
                w = recon_weights.get(f"topo_{k}", 1.0)
                recon_loss = recon_loss + w * F.mse_loss(recons[f"topo_{k}"], x)
                n_present += 1

        for k in self.conn_keys:
            c = connectomes.get(k)
            if c is not None:
                w = recon_weights.get(f"conn_{k}", 1.0)
                recon_loss = recon_loss + w * F.mse_loss(recons[f"conn_{k}"], c)
                n_present += 1

        recon_loss = recon_loss / max(n_present, 1)
        kl_loss    = self.compute_kl(mu, lv)
        total_loss = recon_loss + self.beta * kl_loss

        return {
            "loss":            total_loss,
            "recon_loss":      recon_loss,
            "kl_loss":         kl_loss,
            "z":               z,
            "mu":              mu,
            "lv":              lv,
            "reconstructions": recons,
        }

    # ── inference / translation ───────────────────────────────────────────────

    @torch.no_grad()
    def translate(
        self,
        topomaps:    Optional[dict]   = None,
        connectomes: Optional[dict]   = None,
    ) -> dict:
        """
        Translate from any combination of present modalities to all outputs.

        Examples:
            # Surface → connectome (cross-modal)
            out = model.translate(surface=x_surf)
            conn_pred = out["conn_100"]

            # Connectome → surface (cross-modal)
            out = model.translate(connectomes={"100": c100})
            surf_pred = out["surface"]

            # Joint (best quality)
            out = model.translate(surface=x_surf, connectomes={"100": c100})
        """
        if topomaps is None:
            topomaps = {}
        if connectomes is None:
            connectomes = {}

        topo_params = {
            k: self.encode_surface(topomaps.get(k), k)
            for k in self.topo_keys
        }
        conn_params = {
            k: self.encode_connectome(connectomes.get(k), k)
            for k in self.conn_keys
        }

        z, mu, lv = self.fuse(topo_params, conn_params)
        recons = self.decode_all(z)
        recons["z"] = z
        return recons



