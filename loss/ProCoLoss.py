"""
Probabilistic Contrastive (ProCo) Loss
=======================================
Paper: "Probabilistic Contrastive Learning for Long-Tailed Visual Recognition"
       TPAMI 2024 — Du, Wang, Song, Huang

This file implements:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  vMF distribution  (Section 3.2, Eq. 4–6)                              │
  │  MLE parameter estimation  (Section 3.2, Eq. 8, 10)                    │
  │  Online mean accumulation  (Section 3.2, Eq. 11)                       │
  │  Miller backward recurrence for Bessel function  (Section 3.2, Eq. 22) │
  │  Closed-form ProCo loss  L_in  (Proposition 1, Eq. 13 / Eq. 20)       │
  └─────────────────────────────────────────────────────────────────────────┘

Key equations reproduced verbatim from the paper
-------------------------------------------------
vMF pdf  (Eq. 4):    f_p(z; μ, κ) = exp(κ μᵀz) / C_p(κ)
Normaliser (Eq. 5):  C_p(κ) = (2π)^{p/2} I_{p/2-1}(κ) / κ^{p/2-1}

MLE mean (Eq. 8):    μ_y  = z̄ / R̄
MLE conc (Eq. 10):   κ̂_y  = R̄(p − R̄²) / (1 − R̄²)

Online mean (Eq. 11):
    z̄^(t)_j = [n^(t-1)_j · z̄^(t-1)_j + m^(t)_j · z̄'^(t)_j]
               ─────────────────────────────────────────────────
                          n^(t-1)_j + m^(t)_j

Closed-form loss L_in (Eq. 13 / 20):
    L_ProCo(z_i, y_i) =
        -log[ π_{y_i} · C_p(κ̃_{y_i}) / C_p(κ_{y_i}) ]
        +log[ Σ_j  π_j · C_p(κ̃_j)   / C_p(κ_j)     ]
    where  κ̃_j = ‖ κ_j μ_j + z_i/τ ‖_2

Miller backward recurrence  (Section 3.2, Eq. 22–23):
    I_{ν-1}(κ) = (2ν/κ) I_ν(κ) + I_{ν+1}(κ)   [Eq. 22]
    I_{p/2-1}(κ) = [I_0(κ) / Ĩ_0(κ)] · Ĩ_{p/2-1}(κ)  [Eq. 23]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
#  Bessel Function  I_{p/2 - 1}(κ)
#  Paper Section 3.2 — "we employ the Miller recurrence algorithm [72]"
#  Eq. 22:  I_{ν-1}(κ) = (2ν/κ) I_ν(κ) + I_{ν+1}(κ)
#  Eq. 23:  rescale using known I_0(κ)
# ──────────────────────────────────────────────────────────────────────────────

def bessel_miller(p: int, kappa: torch.Tensor) -> torch.Tensor:
    """
    Compute the modified Bessel function of the first kind I_{p/2-1}(κ)
    using the Miller backward recurrence algorithm.

    Paper Section 3.2:
      "To compute I_{p/2-1}(κ) in ProCo, we follow these steps:
       First, we assign the trial values 1 and 0 to I_M(κ) and I_{M+1}(κ),
       respectively. Here, M is a chosen large positive integer; in our
       experiments, we set M = p."  [Eq. 22–23]

    Falls back to scipy.special.iv when available for better numerical
    stability at the boundary cases.

    Parameters
    ----------
    p     : int           — feature dimension p
    kappa : (N,) tensor   — concentration parameters κ (scalar or batch)

    Returns
    -------
    (N,) tensor  — I_{p/2-1}(kappa)
    """
    target_order = p // 2 - 1   # ν = p/2 − 1  (the order we need)

    # ── scipy path (preferred for accuracy) ─────────────────────────────────
    try:
        import scipy.special as sp
        scalar = kappa.dim() == 0
        if scalar:
            kappa = kappa.unsqueeze(0)
        kappa_np = kappa.detach().cpu().double().numpy()
        iv_np    = sp.iv(target_order, kappa_np)
        iv = torch.from_numpy(iv_np).to(dtype=kappa.dtype, device=kappa.device)
        return iv.squeeze(0) if scalar else iv
    except Exception:
        pass

    # ── Miller backward recurrence  (Eq. 22–23) ─────────────────────────────
    scalar = kappa.dim() == 0
    if scalar:
        kappa = kappa.unsqueeze(0)

    orig_device, orig_dtype = kappa.device, kappa.dtype
    kappa = kappa.to(torch.float64)

    M = p  # starting order M = p  (paper default)

    # Trial initial values: I_M = 1, I_{M+1} = 0  (Section 3.2)
    I_curr = torch.ones_like(kappa)   # I_M
    I_next = torch.zeros_like(kappa)  # I_{M+1}

    I_target_trial = torch.zeros_like(kappa)
    I0_trial       = torch.zeros_like(kappa)

    # Backward recurrence from M down to 0  (Eq. 22)
    for nu in range(M, -1, -1):
        # I_{ν-1}(κ) = (2ν/κ) I_ν(κ) + I_{ν+1}(κ)
        safe_kappa = kappa.clamp(min=1e-10)
        I_prev = (2.0 * nu / safe_kappa) * I_curr + I_next

        # Record trial values at the target order and at order 0
        if nu == target_order + 1:
            I_target_trial = I_curr.clone()
        if nu == 1:
            I0_trial = I_curr.clone()

        I_next = I_curr
        I_curr = I_prev

    # Rescale using exact I_0(κ)  (Eq. 23)
    # I_{p/2-1}(κ) = [I_0(κ) / Ĩ_0(κ)] · Ĩ_{p/2-1}(κ)
    I0_exact = torch.special.i0(kappa.clamp(max=700.0))
    result = (I0_exact / I0_trial.clamp(min=1e-300)) * I_target_trial

    if scalar:
        result = result.squeeze(0)
    return result.to(dtype=orig_dtype, device=orig_device)


# ──────────────────────────────────────────────────────────────────────────────
#  Log-normalising constant  log C_p(κ)
#  Paper Eq. 5:  C_p(κ) = (2π)^{p/2} I_{p/2-1}(κ) / κ^{p/2-1}
#
#  We compute in log-space for numerical stability:
#    log C_p(κ) = (p/2−1)·log κ − (p/2)·log(2π) − log I_{p/2-1}(κ)
# ──────────────────────────────────────────────────────────────────────────────

def log_Cp(p: int, kappa: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable log C_p(κ)  as derived from Eq. 5 of the paper.

    Used in the ProCo loss (Eq. 20) as:
        log[ C_p(κ̃) / C_p(κ) ] = log_Cp(κ̃) − log_Cp(κ)
    """
    kappa = kappa.clamp(min=1e-8)
    log_bessel = torch.log(bessel_miller(p, kappa).clamp(min=1e-30))

    # log C_p(κ) = (p/2 − 1) log κ − (p/2) log(2π) − log I_{p/2−1}(κ)
    val = (p / 2.0 - 1.0) * torch.log(kappa) \
          - (p / 2.0) * math.log(2.0 * math.pi) \
          - log_bessel

    # Clamp to prevent downstream inf/nan in the loss
    return val.clamp(min=-500.0, max=500.0)


# ──────────────────────────────────────────────────────────────────────────────
#  ProCo Loss Module
# ──────────────────────────────────────────────────────────────────────────────

class ProCoLoss(nn.Module):
    """
    Probabilistic Contrastive Loss — L_ProCo = L_in  (Eq. 20).

    The closed-form loss is derived in Proposition 1 (Section 3.2):

        L_ProCo(z_i, y_i) =
            − log[ π_{y_i} · C_p(κ̃_{y_i}) / C_p(κ_{y_i}) ]
            + log[  Σ_j  π_j · C_p(κ̃_j)   / C_p(κ_j)    ]

        where  κ̃_j = ‖ κ_j μ_j + z_i/τ ‖₂

    This is the *expected* SupCon loss as the number of contrastive samples
    N → ∞, which eliminates the need for large batch sizes.

    The implementation uses L_in (not L_out) as the paper states:
        "Since L_in enforces the margin modification as shown in Eq. (1),
        we adopt it as the surrogate loss"  (Section 3.2, Eq. 20)

    Parameters
    ----------
    num_classes : int   — K (number of classes)
    feat_dim    : int   — p (projection head output dimension)
    tau         : float — τ temperature (paper: 0.1 for CIFAR, 0.07 for ImageNet)
    """

    def __init__(self, num_classes: int, feat_dim: int, tau: float = 0.1, device: str = 'cuda'):
        super().__init__()
        self.K   = num_classes
        self.p   = feat_dim
        self.tau = tau       # Paper Section 4.2: τ = 0.1 for CIFAR

        # ── Online estimators for per-class mean direction μ_y  (Eq. 11) ────
        # z_bar      : working estimate used during this epoch's forward pass
        #              = z̄^{(t-1)} from Eq. 11
        # n_count    : n^{(t-1)}_j — samples seen up to last epoch
        # z_bar_new  : accumulator for the current epoch z̄'^{(t)}
        # n_count_new: m^{(t)}_j — samples seen in current epoch so far
        self.register_buffer('z_bar',       torch.zeros(num_classes, feat_dim))
        self.register_buffer('n_count',     torch.zeros(num_classes))
        self.register_buffer('z_bar_new',   torch.zeros(num_classes, feat_dim))
        self.register_buffer('n_count_new', torch.zeros(num_classes))

    # ── Online mean update  (Eq. 11) ────────────────────────────────────────

    @torch.no_grad()
    def update_mean(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Update per-class sample mean from the current mini-batch.

        Paper Eq. 11:
            z̄^(t)_j = [n^(t-1)_j · z̄^(t-1)_j + m^(t)_j · z̄'^(t)_j]
                       ───────────────────────────────────────────────────
                                   n^(t-1)_j + m^(t)_j

        Here we accumulate into z_bar_new / n_count_new during the epoch,
        and the previous-epoch estimate z_bar / n_count is kept frozen
        for the MLE step (so all mini-batches within an epoch share the same
        distribution estimate, mirroring the paper's description).

        features : (B, p) — L2-normalised (unit sphere)
        labels   : (B,)   — integer class indices
        """
        for c in range(self.K):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            z_c = features[mask].float()   # (m, p)
            m   = float(z_c.shape[0])

            # Weighted update of the new-epoch accumulator
            prev_n = self.n_count_new[c].item()
            # z̄ ← (prev_n · z̄ + m · batch_mean) / (prev_n + m)
            self.z_bar_new[c] = (
                prev_n * self.z_bar_new[c] + m * z_c.mean(0)
            ) / (prev_n + m)
            self.n_count_new[c] = prev_n + m

    def end_epoch(self):
        """
        Swap accumulators at the end of each training epoch.

        Paper Section 3.2: "we adopt the estimated sample mean of the
        previous epoch for maximum likelihood estimation, while maintaining
        a new sample mean from zero initialization in the current epoch."
        """
        self.z_bar.copy_(self.z_bar_new)
        self.n_count.copy_(self.n_count_new)
        self.z_bar_new.zero_()
        self.n_count_new.zero_()

    # ── MLE parameter estimation  (Eq. 8, 10) ───────────────────────────────

    def estimate_vmf_params(self):
        """
        Compute MLE estimates (μ_y, κ_y) for all K classes from z̄.

        Paper Eq. 8:   μ_y = z̄ / R̄         (mean direction)
        Paper Eq. 10:  κ̂_y = R̄(p − R̄²) / (1 − R̄²)  (concentration)

        Returns
        -------
        mu    : (K, p) — unit mean directions
        kappa : (K,)   — concentration parameters (clamped ≥ 1e-3)
        """
        z_bar = self.z_bar.clone()                         # (K, p)
        z_bar = torch.nan_to_num(z_bar, nan=0.0)

        R_bar = z_bar.norm(dim=1)                          # (K,)  = ‖z̄‖  (scalar R̄)
        R_bar = R_bar.clamp(min=0.0, max=1.0 - 1e-6)      # keep in (0,1) for Eq.10

        # μ_y = z̄ / R̄  (Eq. 8)
        mu = torch.zeros_like(z_bar)
        valid = R_bar > 1e-8
        if valid.any():
            mu[valid] = z_bar[valid] / R_bar[valid].unsqueeze(1)
        mu = F.normalize(mu, dim=1, eps=1e-8)              # ensure unit norm

        # κ̂ = R̄(p − R̄²) / (1 − R̄²)  (Eq. 10)
        p  = float(self.p)
        R2 = R_bar ** 2
        kappa = R_bar * (p - R2) / (1.0 - R2).clamp(min=1e-8)
        kappa = kappa.clamp(min=1e-3, max=50.0)            # numerical safety

        return mu, kappa

    # ── Forward  (Eq. 20) ───────────────────────────────────────────────────

    def forward(
        self,
        features:    torch.Tensor,   # (B, p)  — L2-normalised projection features
        labels:      torch.Tensor,   # (B,)    — class indices
        class_prior: torch.Tensor,   # (K,)    — π_y = class frequency ratio
    ) -> torch.Tensor:
        """
        Compute L_ProCo = L_in  (Proposition 1, Eq. 13 / Eq. 20).

            L_ProCo(z_i, y_i) =
                -log[ π_{y_i} · C_p(κ̃_{y_i}) / C_p(κ_{y_i}) ]
                +log[  Σ_j  π_j · C_p(κ̃_j)   / C_p(κ_j)    ]

            κ̃_j = ‖ κ_j μ_j + z_i/τ ‖₂

        Parameters
        ----------
        features    : (B, p)  — already L2-normalised
        labels      : (B,)
        class_prior : (K,)    — π_y = N_y / N  (class frequency ratio)
        """
        # Ensure unit-sphere features  (fp16 may drift)
        features = F.normalize(features.float(), dim=1, eps=1e-8)
        features = torch.nan_to_num(features, nan=0.0)

        # Step 1 — Update running mean accumulator  (Eq. 11)
        self.update_mean(features.detach(), labels)

        # Step 2 — MLE estimates of (μ_y, κ_y)  (Eq. 8, 10)
        mu, kappa = self.estimate_vmf_params()             # (K,p), (K,)
        pi        = class_prior.to(features.device)        # (K,)

        B   = features.shape[0]

        # Step 3 — log C_p(κ_j)  for all classes  (Eq. 5 in log-space)
        log_Cp_kappa = log_Cp(self.p, kappa)               # (K,)

        # Step 4 — Compute κ̃_j = ‖ κ_j μ_j + z_i/τ ‖₂  for all (i,j)
        #          κ_j μ_j : (K, p)
        #          z_i/τ   : (B, p)   → broadcast → (B, K, p) → L2 norm → (B, K)
        kappa_mu  = kappa.unsqueeze(1) * mu                # (K, p)
        zi_over_tau = features / self.tau                  # (B, p)

        # (B, K, p) = (1, K, p) + (B, 1, p)
        kappa_tilde = (
            kappa_mu.unsqueeze(0) + zi_over_tau.unsqueeze(1)
        ).norm(dim=2)                                      # (B, K)

        # Step 5 — log C_p(κ̃_j)  for all (B*K,)
        kappa_tilde_flat = kappa_tilde.reshape(-1).clamp(min=1e-4, max=50.0)
        log_Cp_ktilde = log_Cp(self.p, kappa_tilde_flat).reshape(B, self.K)  # (B, K)

        # Step 6 — log[ π_j · C_p(κ̃_j) / C_p(κ_j) ]
        #        = log π_j + log_Cp(κ̃_j) − log_Cp(κ_j)
        log_pi    = torch.log(pi.clamp(min=1e-8))          # (K,)
        log_ratio = (
            log_pi.unsqueeze(0)           # (1, K)
            + log_Cp_ktilde               # (B, K)
            - log_Cp_kappa.unsqueeze(0)   # (1, K)
        )                                                  # (B, K)
        log_ratio = log_ratio.clamp(min=-500.0, max=500.0)

        # Step 7 — L_in  (Eq. 13 / Eq. 20)
        #   numerator   = log[ π_{y_i} · C_p(κ̃_{y_i}) / C_p(κ_{y_i}) ]
        #   denominator = log Σ_j [ π_j · C_p(κ̃_j) / C_p(κ_j) ]
        log_numerator   = log_ratio[torch.arange(B, device=features.device), labels]  # (B,)
        log_denominator = torch.logsumexp(log_ratio, dim=1)                            # (B,)

        loss = (-log_numerator + log_denominator).mean()
        loss = torch.nan_to_num(loss, nan=0.0, posinf=10.0, neginf=-10.0)
        return loss
