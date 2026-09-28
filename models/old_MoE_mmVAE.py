"""
MMVAE: Multimodal Variational Autoencoder (Mixture of Experts)
==============================================================
Based on: Shi et al. (NeurIPS 2019) - "Variational Mixture-of-Experts
Autoencoders for Multi-Modal Deep Generative Models"
https://arxiv.org/abs/1911.03393

Core idea:
    Instead of multiplying expert Gaussians (PoE), the MMVAE forms the
    joint posterior as a *mixture* (MoE):

        q(z | x_1,...,x_M) = (1/M) * sum_m q(z | x_m)

    Crucially, this mixture cannot be sampled directly. Instead, the MMVAE
    uses a Mixture of Experts importance-weighted estimator (IWAE-style):

        For each modality m:
            1. Sample z_m ~ q(z | x_m)          ← unimodal posterior
            2. Compute cross-reconstruction of ALL other modalities from z_m
            3. Compute log-weight w_m = log p(x | z_m) - log q(z_m | x_m)

    The ELBO is the average over these per-modality sub-ELBOs.

    Key differences vs. MVAE (PoE):
        PoE  → sharp joint posterior, strong cross-modal coherence,
                but mode-seeking (can ignore weak modalities).
        MoE  → softer joint posterior, better coverage of each modality's
                unimodal structure, easier to handle missing modalities
                without the PoE collapsing to the prior expert.

Usage:
    - Subclass ModalityEncoder / ModalityDecoder (same interface as mvae.py).
    - Pass lists of encoders/decoders to MMVAE.
    - Forward pass accepts list[Optional[Tensor]]; None = missing modality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base classes  (identical interface to mvae.py for easy swapping)
# ─────────────────────────────────────────────────────────────────────────────

class ModalityEncoder(nn.Module):
    """
    Returns (mu, log_var) of shape (batch, latent_dim).
    """
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class ModalityDecoder(nn.Module):
    """
    Takes z of shape (batch, latent_dim), returns reconstruction.
    """
    def forward(self, z: Tensor) -> Tensor:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Reparameterisation  (shared utility)
# ─────────────────────────────────────────────────────────────────────────────

def reparameterise(mu: Tensor, log_var: Tensor) -> Tensor:
    """z = mu + eps * std,  eps ~ N(0, I)"""
    if not torch.is_grad_enabled():
        return mu
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


# ─────────────────────────────────────────────────────────────────────────────
# Log-probability helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_prob_gaussian(x: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
    """
    Element-wise log N(x | mu, exp(log_var)).
    Returns (batch,) — summed over the feature dimension.
    """
    # -0.5 * [log(2pi) + log_var + (x - mu)^2 / var]
    log2pi   = torch.log(torch.tensor(2.0 * torch.pi, device=x.device))
    per_dim  = -0.5 * (log2pi + log_var + (x - mu).pow(2) / (log_var.exp() + 1e-8))
    return per_dim.sum(dim=-1)                                 # (batch,)


def log_prob_bernoulli(x: Tensor, logits: Tensor) -> Tensor:
    """
    Binary cross-entropy log-likelihood (Bernoulli decoder).
    Expects raw logits from the decoder.
    Returns (batch,) — summed over feature dimension.
    """
    return -F.binary_cross_entropy_with_logits(
        logits, x, reduction="none"
    ).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# MMVAE
# ─────────────────────────────────────────────────────────────────────────────

class MMVAE(nn.Module):
    """
    Multimodal VAE using Mixture of Experts (MoE) posterior.

    Args:
        encoders:       list[ModalityEncoder] — one per modality
        decoders:       list[ModalityDecoder] — one per modality (same order)
        latent_dim:     dimensionality of the shared latent space
        beta:           KL weight (beta-VAE; default 1.0)
        likelihood:     'gaussian' or 'bernoulli' — decoder likelihood type
        n_iwae_samples: number of importance-weighted samples per modality
                        (K in the IWAE estimator; 1 = standard ELBO)

    Forward:
        inputs: list[Optional[Tensor]] — one per modality, None if absent.
        Returns a dict with latent codes, reconstructions, and per-modality
        posterior parameters.

    Training loss:
        Call mmvae.compute_loss(inputs) to get the MoE ELBO directly.
        The returned dict has 'loss', 'recon_loss', 'kl_loss'.
    """

    def __init__(
        self,
        encoders: list[ModalityEncoder],
        decoders: list[ModalityDecoder],
        latent_dim: int,
        beta: float = 1.0,
        likelihood: str = "gaussian",
        n_iwae_samples: int = 1,
    ):
        super().__init__()
        assert len(encoders) == len(decoders), \
            "Must provide one decoder per encoder."
        assert likelihood in ("gaussian", "bernoulli"), \
            "likelihood must be 'gaussian' or 'bernoulli'."

        self.encoders        = nn.ModuleList(encoders)
        self.decoders        = nn.ModuleList(decoders)
        self.latent_dim      = latent_dim
        self.beta            = beta
        self.likelihood      = likelihood
        self.n_iwae_samples  = n_iwae_samples
        self.n_modalities    = len(encoders)

    # ── Per-modality encoding ─────────────────────────────────────────────────

    def encode_all(
        self,
        inputs: list[Optional[Tensor]]
    ) -> list[Optional[tuple[Tensor, Tensor]]]:
        """
        Encode each present modality independently.

        Returns:
            list of (mu_m, log_var_m) or None for missing modalities.
            Each tensor is (batch, latent_dim).
        """
        params = []
        for encoder, x in zip(self.encoders, inputs):
            if x is None:
                params.append(None)
            else:
                mu_m, lv_m = encoder(x)
                params.append((mu_m, lv_m))
        return params

    # ── Decoding ──────────────────────────────────────────────────────────────

    def decode_all(self, z: Tensor) -> list[Tensor]:
        """Decode a shared z into reconstructions for all modalities."""
        return [decoder(z) for decoder in self.decoders]

    # ── KL divergence ─────────────────────────────────────────────────────────

    def kl_divergence(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Analytical KL[ N(mu, exp(log_var)) || N(0, I) ]
        Returns scalar (mean over batch and latent dims).
        """
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        return kl.sum(dim=-1).mean()

    # ── MoE ELBO (core of MMVAE) ──────────────────────────────────────────────

    def moe_elbo(
        self,
        inputs: list[Optional[Tensor]],
        modality_params: list[Optional[tuple[Tensor, Tensor]]]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute the Mixture-of-Experts ELBO.

        Algorithm (per modality m that is present):
            1. Sample z_m ~ q(z | x_m)  [reparameterise]
            2. Compute log p(x_i | z_m) for all present modalities x_i
               → cross-modal reconstruction
            3. Compute log q(z_m | x_m) — the encoder log-prob
            4. sub_ELBO_m = sum_i log p(x_i | z_m) - KL_m

        Final ELBO = mean over present modalities of sub_ELBO_m.

        With n_iwae_samples > 1 the estimator uses multiple z samples per
        modality and averages their log-weights (IWAE-style).

        Returns:
            total_loss, recon_loss, kl_loss — all scalars
        """
        present_indices = [
            i for i, p in enumerate(modality_params) if p is not None
        ]
        if len(present_indices) == 0:
            raise ValueError("At least one modality must be present.")

        device = modality_params[present_indices[0]][0].device
        total_recon = torch.tensor(0.0, device=device)
        total_kl    = torch.tensor(0.0, device=device)

        for m in present_indices:
            mu_m, lv_m = modality_params[m]

            # Accumulate IWAE estimates over K samples
            lw_samples = []
            for _ in range(self.n_iwae_samples):
                z_m = reparameterise(mu_m, lv_m)              # (batch, latent)

                # ── Cross reconstruction: all present modalities from z_m ──────
                recons_m     = self.decode_all(z_m)
                log_px_given_z = torch.tensor(0.0, device=device)

                for i in present_indices:
                    x_i     = inputs[i]
                    recon_i = recons_m[i]

                    if self.likelihood == "gaussian":
                        # Treat decoder output as mu; log_var=0 (unit variance)
                        log_var_out = torch.zeros_like(recon_i)
                        lp = log_prob_gaussian(x_i, recon_i, log_var_out)
                    else:
                        lp = log_prob_bernoulli(x_i, recon_i)

                    log_px_given_z = log_px_given_z + lp.mean()

                # ── log q(z_m | x_m) under the m-th encoder ──────────────────
                log_qz = log_prob_gaussian(z_m, mu_m, lv_m).mean()

                # ── log p(z) = log N(0, I) ────────────────────────────────────
                log_pz = log_prob_gaussian(
                    z_m,
                    torch.zeros_like(z_m),
                    torch.zeros_like(z_m)    # log_var=0 → var=1
                ).mean()

                lw_samples.append(log_px_given_z + log_pz - log_qz)

            # Average log-weights across IWAE samples
            lw_mean     = torch.stack(lw_samples).mean()
            total_recon = total_recon + (lw_mean + self.kl_divergence(mu_m, lv_m))
            total_kl    = total_kl    + self.kl_divergence(mu_m, lv_m)

        n_present   = len(present_indices)
        recon_loss  = -total_recon / n_present             # negative because we minimise
        kl_loss     = total_kl     / n_present
        total_loss  = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    # ── compute_loss convenience wrapper ─────────────────────────────────────

    def compute_loss(
        self,
        inputs: list[Optional[Tensor]]
    ) -> dict:
        """
        Full forward + MoE ELBO loss computation in one call.
        Use this during training.

        Returns dict with keys:
            loss, recon_loss, kl_loss, modality_params
        """
        modality_params = self.encode_all(inputs)
        total, recon, kl = self.moe_elbo(inputs, modality_params)
        return {
            "loss":            total,
            "recon_loss":      recon,
            "kl_loss":         kl,
            "modality_params": modality_params,
        }

    # ── Forward (returns latents + reconstructions) ───────────────────────────

    def forward(
        self,
        inputs: list[Optional[Tensor]]
    ) -> dict:
        """
        Inference forward pass.

        Samples z independently from each present modality encoder,
        then decodes all modalities from each z.

        Returns dict with:
            z_per_modality    — list of (batch, latent) or None if modality absent
            reconstructions   — list of lists: recons[m][i] = decode_i(z_m)
            modality_params   — list of (mu_m, log_var_m) or None
            z_mixture         — single z sampled from a randomly chosen
                                present modality (uniform MoE sample)
            recon_from_mixture — decode_all(z_mixture)
        """
        modality_params = self.encode_all(inputs)
        present_indices = [i for i, p in enumerate(modality_params) if p is not None]

        z_per_modality  = []
        reconstructions = []

        for i, params in enumerate(modality_params):
            if params is None:
                z_per_modality.append(None)
                reconstructions.append(None)
            else:
                mu_i, lv_i = params
                z_i = reparameterise(mu_i, lv_i)
                z_per_modality.append(z_i)
                reconstructions.append(self.decode_all(z_i))

        # Sample from the mixture: pick a random present modality (uniform MoE)
        chosen_m = present_indices[torch.randint(len(present_indices), (1,)).item()]
        z_mixture = z_per_modality[chosen_m]
        recon_from_mixture = self.decode_all(z_mixture)

        return {
            "z_per_modality":      z_per_modality,
            "reconstructions":     reconstructions,
            "modality_params":     modality_params,
            "z_mixture":           z_mixture,
            "recon_from_mixture":  recon_from_mixture,
        }

    # ── Joint posterior mean (for downstream tasks) ───────────────────────────

    def joint_representation(
        self,
        inputs: list[Optional[Tensor]]
    ) -> Tensor:
        """
        Returns a single joint latent by averaging the per-modality means.
        This is a deterministic summary useful for classification or
        regression heads — it approximates the MoE mean.
        """
        modality_params = self.encode_all(inputs)
        mus = [p[0] for p in modality_params if p is not None]
        return torch.stack(mus, dim=0).mean(dim=0)   # (batch, latent_dim)

    # ── Sample from prior ─────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> list[Tensor]:
        """Sample n_samples from N(0,I) and decode all modalities."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode_all(z)


# ─────────────────────────────────────────────────────────────────────────────
# Example encoders / decoders (MLP-based) — replace with your transformers
# ─────────────────────────────────────────────────────────────────────────────

class MLPEncoder(ModalityEncoder):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_log_var(h)


class MLPDecoder(ModalityDecoder):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    LATENT = 32
    BATCH  = 8
    DIM_A  = 128   # e.g. EEG features
    DIM_B  = 256   # e.g. fMRI features

    encoders = [MLPEncoder(DIM_A, 64, LATENT), MLPEncoder(DIM_B, 128, LATENT)]
    decoders = [MLPDecoder(LATENT, 64, DIM_A), MLPDecoder(LATENT, 128, DIM_B)]

    model = MMVAE(
        encoders, decoders,
        latent_dim=LATENT,
        beta=1.0,
        likelihood="gaussian",
        n_iwae_samples=5,
    )

    x_a = torch.randn(BATCH, DIM_A)
    x_b = torch.randn(BATCH, DIM_B)

    # ── Both modalities ────────────────────────────────────────────────────────
    losses = model.compute_loss([x_a, x_b])
    print(f"[Both]   loss={losses['loss']:.4f}  "
          f"recon={losses['recon_loss']:.4f}  kl={losses['kl_loss']:.4f}")

    out = model([x_a, x_b])
    print(f"[Both]   z_mixture: {out['z_mixture'].shape}")
    print(f"[Both]   recon shapes: {[r.shape for r in out['recon_from_mixture']]}")

    # ── Missing modality B ─────────────────────────────────────────────────────
    losses_missing = model.compute_loss([x_a, None])
    print(f"[A only] loss={losses_missing['loss']:.4f}  "
          f"recon={losses_missing['recon_loss']:.4f}  kl={losses_missing['kl_loss']:.4f}")

    # ── Joint representation (for downstream tasks) ────────────────────────────
    z_joint = model.joint_representation([x_a, x_b])
    print(f"[Both]   z_joint: {z_joint.shape}")

    # ── Generation from prior ──────────────────────────────────────────────────
    samples = model.sample(4, device=torch.device("cpu"))
    print(f"[Sample] recon shapes: {[s.shape for s in samples]}")
