"""
ResNet-32 with ProCo Two-Branch Design
=======================================
Paper: "Probabilistic Contrastive Learning for Long-Tailed Visual Recognition"
       TPAMI 2024 — Du, Wang, Song, Huang

Architecture (Section 3.2 — Overall Objective):
  ┌─────────────────────────────────────────────────────────┐
  │  Input image x                                          │
  │      │                                                  │
  │      ▼                                                  │
  │  Backbone  F(·)  — ResNet-32  [He et al. CVPR 2016]    │
  │      │                                                  │
  │      ├──────────────────────┐                           │
  │      ▼                      ▼                           │
  │  Classifier G(·)    Projection Head P(·)               │
  │  linear, K outputs  MLP → L2-normalised                │
  │      │                      │                           │
  │      ▼                      ▼                           │
  │  logits (B,K)         z_proj (B,p)                     │
  │      │                      │                           │
  │  L_LA loss            L_ProCo loss                     │
  └─────────────────────────────────────────────────────────┘

Paper specs (Section 4.2, CIFAR setting):
  Backbone      : ResNet-32
  Classifier    : linear layer (cosine classifier in paper: normalised linear)
  Projection    : 2-layer MLP, hidden=512, output=128, L2-normalised output
  Temperature   : τ = 0.1  (for ProCo loss, CIFAR)
  Overall loss  : L = L_LA + α · L_ProCo,  α = 1.0  (Eq. 24)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
#  ResNet-32 Backbone  (CIFAR variant)
#  Paper Section 4.2: "We adopt ResNet-32 [2] as the backbone network"
#  He et al. "Deep residual learning for image recognition" CVPR 2016  [2]
# ──────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """
    Standard BasicBlock (no bottleneck) for ResNet-32.
    Each block: Conv3×3 → BN → ReLU → Conv3×3 → BN + skip → ReLU
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # Shortcut connection: 1×1 conv + BN if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet32(nn.Module):
    """
    ResNet-32 for CIFAR (32×32 images).

    Architecture:
      conv1 (3→16, 3×3)
      layer1: 5 × BasicBlock(16, 16, stride=1)
      layer2: 5 × BasicBlock(16, 32, stride=2)
      layer3: 5 × BasicBlock(32, 64, stride=2)
      global avg-pool → 64-d feature vector

    Total depth: 1 + 3 stages × 5 blocks × 2 layers + 1 = 32  ✓
    """

    def __init__(self):
        super().__init__()
        self.in_planes = 16

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        # Three stages  (He et al. Table 1 for CIFAR ResNet-32)
        self.layer1 = self._make_layer(16, num_blocks=5, stride=1)
        self.layer2 = self._make_layer(32, num_blocks=5, stride=2)
        self.layer3 = self._make_layer(64, num_blocks=5, stride=2)

        self.feat_dim = 64   # output feature dimension

        # Weight initialisation (standard for ResNets)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)   # (B,16,32,32)
        out = self.layer1(out)                                 # (B,16,32,32)
        out = self.layer2(out)                                 # (B,32,16,16)
        out = self.layer3(out)                                 # (B,64, 8, 8)
        out = F.adaptive_avg_pool2d(out, 1)                   # (B,64, 1, 1)
        return out.view(out.size(0), -1)                      # (B,64)


# ──────────────────────────────────────────────────────────────────────────────
#  MLP Projection Head  P(·)
#  Paper Section 3.2: "a projection head P(·), which is an MLP that maps
#  the representation to another feature space for decoupling with the
#  classifier"
#  Paper Section 4.2 (CIFAR): hidden=512, output_dim=128
# ──────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head with L2-normalised output.

    Paper CIFAR settings (Section 4.2):
        hidden_dim = 512
        out_dim    = 128

    The output is L2-normalised so that features lie on the unit
    hypersphere S^{p-1}, matching the vMF distribution assumption.

    Architecture: Linear(in→hidden) → BN → ReLU → Linear(hidden→out) → L2-norm
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=1, eps=1e-8)   # → unit hypersphere


# ──────────────────────────────────────────────────────────────────────────────
#  Full ProCo Model  (Two-branch design)
#  Paper Section 3.2 — Overall Objective, Eq. 24:
#      L = L_LA + α · L_ProCo
# ──────────────────────────────────────────────────────────────────────────────

class ProCoModel(nn.Module):
    """
    Shared backbone + classification branch + representation branch.

    Classification branch  G(·): linear classifier → logits (B, K)
                                  optimised with L_LA  (Eq. 24)

    Representation branch  P(·): MLP projection head → z_proj (B, p)
                                  optimised with L_ProCo  (Eq. 24)

    Forward outputs:
        logits  : (B, K)   — raw logits for L_LA
        z_proj  : (B, p)   — L2-normalised features for L_ProCo
    """

    def __init__(
        self,
        num_classes: int = 100,
        proj_hidden: int = 512,    # Paper CIFAR default: 512
        proj_out:    int = 128,    # Paper CIFAR default: 128  (= p, feat_dim for ProCo)
    ):
        super().__init__()

        # Shared backbone F(·)
        self.backbone = ResNet32()
        feat_dim      = self.backbone.feat_dim   # 64

        # Classification branch G(·): linear layer
        # Paper uses "a linear classifier G(·)"  (Section 3.2)
        # For ImageNet the paper uses a cosine classifier, but for CIFAR
        # experiments the standard linear classifier is used.
        self.classifier = nn.Linear(feat_dim, num_classes, bias=True)

        # Representation branch P(·): MLP projection head
        self.proj_head = ProjectionHead(
            in_dim=feat_dim,
            hidden_dim=proj_hidden,
            out_dim=proj_out,
        )

        self.feat_dim = feat_dim
        self.proj_dim = proj_out

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 3, 32, 32)

        Returns
        -------
        logits : (B, K)   — classification logits
        z_proj : (B, p)   — L2-normalised projection features
        """
        h      = self.backbone(x)       # (B, 64)
        logits = self.classifier(h)     # (B, K)
        z_proj = self.proj_head(h)      # (B, p=128), unit norm
        return logits, z_proj

    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features only (useful for feature-based evaluation)."""
        return self.backbone(x)
