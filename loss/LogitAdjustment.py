"""
Logit Adjustment Loss  —  Classification Branch Loss L_LA
==========================================================
Paper: "Probabilistic Contrastive Learning for Long-Tailed Visual Recognition"
       TPAMI 2024 — Du, Wang, Song, Huang

This is the classification branch loss used in ProCo's two-branch design.
It is taken from:
  Menon et al. "Long-tail learning via logit adjustment" ICLR 2021  [13]

Paper Eq. 1:
    L_LA(x_i, y_i) = -log [ π_{y_i} · exp(φ_{y_i}(x))
                             ─────────────────────────────────── ]
                             Σ_{y'} π_{y'} · exp(φ_{y'}(x))

Which is equivalent to cross-entropy after adding log(π_y) to each logit:
    adjusted_logit_y = φ_y(x) + log(π_y)

Paper Section 3.1: "Logit Adjustment [13] is a loss margin modification
method. It adopts the prior probability of each class as the margin during
the training and inference process."

Paper Section 3.2 (Overall Objective):
    L = L_LA + α · L_ProCo                              (Eq. 24)

where α is the representation branch weight (paper default α = 1.0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustmentLoss(nn.Module):
    """
    Cross-entropy loss with logit adjustment by log-class-prior.

    Paper Eq. 1:
        L_LA(x_i, y_i) = CE(φ(x) + log π, y_i)

    Parameters
    ----------
    class_freq : (K,) tensor — raw class sample counts N_y
                 (will be normalised to π_y = N_y / N internally)
    tau        : float — scaling of log-prior adjustment
                 Paper uses τ_LA = 1.0  (default, Section 3.1)
    """

    def __init__(self, class_freq: torch.Tensor, tau: float = 1.0):
        super().__init__()
        self.tau = tau

        # Compute π_y = N_y / N  (class prior)
        pi = class_freq.float() / class_freq.float().sum()

        # Store log π_y as buffer so it moves with .to(device)
        self.register_buffer('log_pi', torch.log(pi.clamp(min=1e-8)))   # (K,)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, K)   — raw classifier output φ(x)
        labels : (B,)     — ground-truth class indices

        Returns
        -------
        scalar loss
        """
        # Adjust logits: φ_y(x) + τ · log(π_y)     (Eq. 1)
        adjusted = logits + self.tau * self.log_pi.unsqueeze(0)   # (B, K)
        return F.cross_entropy(adjusted, labels)
