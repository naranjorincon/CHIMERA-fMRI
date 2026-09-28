"""
moe_model.py
============
Mixture-of-Experts multimodal fusion for SiT surface + connectome data.

Key difference from PoE:
    PoE multiplies expert posteriors → sharp joint posterior, mode-seeking.
    MoE averages expert posteriors  → softer joint, mean-seeking, more
    forgiving when expert embeddings encode partially different variance.

The MMVAE loss (Shi et al. 2019) trains each expert's sub-ELBO separately:
    For each present modality m:
        1. Sample z_m ~ q(z | x_m)
        2. Reconstruct ALL present modalities from z_m
        3. Loss_m = sum_i MSE(x_i, decode_i(z_m)) + beta * KL(q_m || p)
    Total loss = mean over m of Loss_m

This means every expert is trained to produce a z that decodes all modalities,
not just its own — which is what forces the shared space to emerge.

Same interface as PoEMultimodalModel so you can swap them directly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import warnings

from models.experts import (
    FrozenSiTExpert,
    FrozenConnectomeExpert,
    SurfaceDecoder,
    ConnectomeDecoder,
    mixture_of_experts,
    reparameterise,
    LOG_VAR_MIN, LOG_VAR_MAX, EPS,
)


class MoEMultimodalModel(nn.Module):
    """
    Mixture-of-Experts fusion of SiT surface + connectome embeddings.

    Additional arg vs PoE:
        n_iwae_samples : number of z samples per expert for the MoE ELBO.
                         1 = standard ELBO, >1 = tighter IWAE bound.
                         Start with 1; increase to 5 once training is stable.
    """

    def __init__(
        self,
        sit_encoders:  dict[str, nn.Module], #nn.Module,
        topo_networks: dict[str, int],
        conn_encoders: dict[str, nn.Module],
        conn_latents:  dict[str, int],
        conn_n_tri:    dict[str, int],
        fusion_dim:    int   = 128,
        sit_latent:    int   = 192,
        pool_patches:  bool  = True,
        beta:          float = 0.0,
        free_bits:     float = 0.1,
        conn_hidden:   int   = 512,
        n_iwae_samples: int  = 1,
        ):
        super().__init__()
        self.fusion_dim     = fusion_dim
        self.pool_patches   = pool_patches
        self.beta           = beta
        self.free_bits      = free_bits
        self.n_iwae_samples = n_iwae_samples
        self.conn_keys      = list(conn_encoders.keys())
        self.topo_keys      = list(sit_encoders.keys())

        # ── frozen experts ────────────────────────────────────────────────────
        # self.surface_expert = FrozenSiTExpert(
        #     sit_encoder,
        #     sit_latent=sit_latent,
        #     fusion_dim=fusion_dim,
        #     pool_patches=pool_patches,
        # )
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

        # ── decoders ──────────────────────────────────────────────────────────
        # self.surface_decoder = SurfaceDecoder(
        #     fusion_dim=fusion_dim,
        #     pool_patches=pool_patches,
        # )
        self.topo_decoders = nn.ModuleDict({
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

    # ── encode ────────────────────────────────────────────────────────────────
    def _encode_all(
        self,
        topomaps: dict[str, Optional[Tensor]], #Optional[Tensor],
        connectomes: dict[str, Optional[Tensor]],
    ) -> dict[str, Optional[tuple]]:
        """
        Returns a dict of all per-modality (mu, lv) params.
        Key "surface" for surface, "conn_100" etc for connectomes.
        Value is None if that modality was not provided.
        """
        params = {}

        # params["surface"] = (
        #     self.surface_expert(surface) if surface is not None else None
        # )
        for k in self.topo_keys:
            c = topomaps.get(k)
            params[f"topo_{k}"] = (
                self.surface_experts[k](c) if c is not None else None
            )
        for k in self.conn_keys:
            c = connectomes.get(k)
            params[f"conn_{k}"] = (
                self.conn_experts[k](c) if c is not None else None
            )
        return params

    # ── decode ────────────────────────────────────────────────────────────────

    def _decode_all(self, z: Tensor) -> dict[str, Tensor]:
        # out = {"surface": self.surface_decoder(z)}
        out = {}
        for k in self.topo_keys:
            out[f"topo_{k}"] = self.topo_decoders[k](z)

        for k in self.conn_keys:
            out[f"conn_{k}"] = self.conn_decoders[k](z)
        return out

    # ── KL divergence ─────────────────────────────────────────────────────────

    def _kl(self, mu: Tensor, lv: Tensor) -> Tensor:
        lv_s = lv.clamp(LOG_VAR_MIN, LOG_VAR_MAX)
        kl = -0.5 * (1 + lv_s - mu.pow(2) - lv_s.exp())
        if self.free_bits > 0:
            kl = kl.clamp(min=self.free_bits)
        return kl.clamp(max=200.0).sum(dim=-1).mean()

    # ── MoE ELBO ─────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        topomaps:     dict[str, Optional[Tensor]], #Optional[Tensor],
        connectomes:  dict[str, Optional[Tensor]],
        recon_weights: Optional[dict] = None,
    ) -> dict:
        """
        MoE ELBO loss.

        For each present modality m:
            Sample z_m ~ q(z | x_m)   [n_iwae_samples times]
            Reconstruct ALL present modalities from z_m
            sub_loss_m = mean_i MSE(x_i, decode_i(z_m)) + beta * KL_m

        Total loss = mean over m of sub_loss_m.

        This forces every expert's latent to be decodable into all modalities.
        """
        if recon_weights is None:
            recon_weights = {}
        if topomaps is None:
            topomaps = {}
        if connectomes is None:
            connectomes = {}

        # Encode all present modalities
        all_params = self._encode_all(topomaps, connectomes)
        # Build list of (modality_key, mu, lv, original_tensor) for present ones
        present = []
        for k in self.topo_keys:
            c = topomaps.get(k)
            p = all_params.get(f"topo_{k}")
            if c is not None and p is not None:
                mu, lv = p
                present.append((f"topo_{k}", mu, lv, c))

        for k in self.conn_keys:
            c = connectomes.get(k)
            p = all_params.get(f"conn_{k}")
            if c is not None and p is not None:
                mu, lv = p
                present.append((f"conn_{k}", mu, lv, c))

        if len(present) == 0:
            raise ValueError("At least one modality must be provided.")

        device     = present[0][1].device
        total_loss = torch.tensor(0.0, device=device)
        total_recon = torch.tensor(0.0, device=device)
        total_kl    = torch.tensor(0.0, device=device)

        # ── per-expert sub-ELBO ───────────────────────────────────────────────
        for mod_key, mu_m, lv_m, _ in present:

            sub_recon = torch.tensor(0.0, device=device)

            for _ in range(self.n_iwae_samples):
                z_m = reparameterise(mu_m, lv_m)
                recons_m = self._decode_all(z_m)

                # Cross-reconstruct all present modalities from z_m
                n_recon = 0
                for tgt_key, _, _, x_tgt in present:
                    w = recon_weights.get(tgt_key, 1.0)
                    sub_recon = sub_recon + w * F.mse_loss(recons_m[tgt_key], x_tgt)
                    n_recon += 1

                sub_recon = sub_recon / max(n_recon, 1)

            sub_recon  = sub_recon / self.n_iwae_samples
            sub_kl     = self._kl(mu_m, lv_m)
            sub_total  = sub_recon + self.beta * sub_kl

            total_loss  = total_loss  + sub_total
            total_recon = total_recon + sub_recon
            total_kl    = total_kl    + sub_kl

        n_experts   = len(present)
        total_loss  = total_loss  / n_experts
        total_recon = total_recon / n_experts
        total_kl    = total_kl    / n_experts

        # ── also compute MoE mixture z for logging / inference ─────────────────
        mu_list = [p[1] for p in present]
        lv_list = [p[2] for p in present]
        mu_moe, lv_moe = mixture_of_experts(mu_list, lv_list)
        z_moe = reparameterise(mu_moe, lv_moe)

        return {
            "loss":            total_loss,
            "recon_loss":      total_recon,
            "kl_loss":         total_kl,
            "z":               z_moe,
            "mu":              mu_moe,
            "lv":              lv_moe,
            "reconstructions": self._decode_all(z_moe),
        }

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def translate(
        self,
        topomaps:    Optional[dict]   = None, #Optional[Tensor] = None,
        connectomes: Optional[dict]   = None,
    ) -> dict:
        """
        Same interface as PoEMultimodalModel.translate().
        At inference, samples z from the MoE mixture posterior.
        """
        if connectomes is None:
            connectomes = {}
        if topomaps is None:
            topomaps = {}

        all_params = self._encode_all(topomaps, connectomes)

        mu_list, lv_list = [], []
        for p in all_params.values():
            if p is not None:
                mu_list.append(p[0])
                lv_list.append(p[1])

        if len(mu_list) == 0:
            raise ValueError("At least one modality must be provided.")

        mu_moe, lv_moe = mixture_of_experts(mu_list, lv_list)
        z = reparameterise(mu_moe, lv_moe)

        recons = self._decode_all(z)
        recons["z"] = z
        return recons
