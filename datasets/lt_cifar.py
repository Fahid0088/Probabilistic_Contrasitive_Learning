"""
Long-Tailed CIFAR-10 / CIFAR-100 Dataset
=========================================
Paper Section 4.1 — Dataset and Evaluation Protocol
Paper Section 4.2 — Implementation Details (CIFAR-10/100-LT)

Exponential sampling: N_j = N * lambda^j,  lambda in (0,1)
Imbalance factor:     gamma = max(N_j) / min(N_j)

Paper splits (Section 4.1):
  Many-shot  : > 100 images per class
  Medium-shot: 20 – 100 images per class
  Few-shot   : < 20 images per class

Augmentation (Section 4.2, CIFAR):
  Classification branch : AutoAugment + Cutout   [paper: "AutoAug and Cutout"]
  Representation branch : SimAugment              [paper: "SimAug"]
  Validation            : ToTensor + Normalize only
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
#  Cutout Augmentation (paper Section 4.2 — used with AutoAugment)
# ──────────────────────────────────────────────────────────────────────────────

class Cutout:
    """
    Randomly mask a square patch of side `length` pixels.
    Paper uses Cutout after AutoAugment for the classification branch.
    Reference: DeVries & Taylor (2017), used by BCL [16] and ProCo.
    """
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes  # paper default: 1 hole
        self.length  = length   # paper default: 16 pixels for CIFAR

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        mask = np.ones((h, w), dtype=np.float32)
        for _ in range(self.n_holes):
            cy = np.random.randint(h)
            cx = np.random.randint(w)
            y1 = max(0, cy - self.length // 2)
            y2 = min(h, cy + self.length // 2)
            x1 = max(0, cx - self.length // 2)
            x2 = min(w, cx + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


# ──────────────────────────────────────────────────────────────────────────────
#  Normalisation constants (standard for CIFAR)
# ──────────────────────────────────────────────────────────────────────────────

_CIFAR_MEAN = {
    'cifar10':  (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}
_CIFAR_STD = {
    'cifar10':  (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}


# ──────────────────────────────────────────────────────────────────────────────
#  Classification-branch transforms  (AutoAugment + Cutout)
#  Paper Section 4.2: "we apply AutoAug and Cutout as the data augmentation
#  strategies for the classification branch"
# ──────────────────────────────────────────────────────────────────────────────

def get_cls_transforms(dataset: str = 'cifar100') -> transforms.Compose:
    """
    Transforms for the classification branch.
    Paper: AutoAugment + Cutout  (Section 4.2)
    """
    mean = _CIFAR_MEAN[dataset]
    std  = _CIFAR_STD[dataset]
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # AutoAugment — paper cites [85] Cubuk et al. CVPR 2019
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # Cutout — paper cites [86] DeVries & Taylor 2017
        Cutout(n_holes=1, length=16),
    ])


# ──────────────────────────────────────────────────────────────────────────────
#  Representation-branch transforms  (SimAugment / SimAug)
#  Paper Section 4.2: "SimAug for the representation branch"
#  SimAug = random crop + color jitter + grayscale + horizontal flip + blur
#  (Chen et al. [51], SimCLR augmentation pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def get_repr_transforms(dataset: str = 'cifar100') -> transforms.Compose:
    """
    Transforms for the representation branch.
    Paper: SimAugment [51] (Chen et al. "A simple framework for contrastive
    learning of visual representations", ICML 2020)
    """
    mean = _CIFAR_MEAN[dataset]
    std  = _CIFAR_STD[dataset]
    s = 0.5  # color jitter strength — standard for CIFAR SimAug
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ──────────────────────────────────────────────────────────────────────────────
#  Validation transforms  (no augmentation)
# ──────────────────────────────────────────────────────────────────────────────

def get_val_transforms(dataset: str = 'cifar100') -> transforms.Compose:
    mean = _CIFAR_MEAN[dataset]
    std  = _CIFAR_STD[dataset]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# keep old name for backward compat
def get_transforms(split: str = 'train', dataset: str = 'cifar100'):
    if split == 'train':
        return get_cls_transforms(dataset)
    return get_val_transforms(dataset)

def get_train_transforms_with_cutout(dataset: str = 'cifar100'):
    return get_cls_transforms(dataset)


# ──────────────────────────────────────────────────────────────────────────────
#  Long-Tailed CIFAR Dataset
#  Paper Section 4.1 — exponential sub-sampling
# ──────────────────────────────────────────────────────────────────────────────

class LongTailedCIFAR(Dataset):
    """
    Long-tailed variant of CIFAR-10/100.

    Sampling rule (Section 4.1, Eq. in text):
        N_j = N * lambda^j,   j = 0, 1, ..., K-1
        lambda = (1 / gamma) ^ {1 / (K-1)}
        gamma  = max(N_j) / min(N_j)    [imbalance factor]

    After sub-sampling the validation set is left BALANCED (original).
    The paper reports Top-1 accuracy on the original balanced val split.

    Paper gamma values tested: 10, 50, 100  (Section 4.2)
    """

    def __init__(
        self,
        root:             str   = './data',
        dataset:          str   = 'cifar100',
        imbalance_factor: int   = 100,          # gamma: 10 / 50 / 100
        train:            bool  = True,
        transform               = None,
        download:         bool  = True,
    ):
        self.transform = transform
        self.train     = train

        # ── Load base CIFAR dataset ──────────────────────────────────────────
        if dataset == 'cifar10':
            base = torchvision.datasets.CIFAR10(
                root=root, train=train, download=download)
            self.num_classes = 10
        else:
            base = torchvision.datasets.CIFAR100(
                root=root, train=train, download=download)
            self.num_classes = 100

        data   = np.array(base.data)      # (N, 32, 32, 3)  uint8
        labels = np.array(base.targets)   # (N,)             int

        # ── Apply long-tail sub-sampling (train only) ────────────────────────
        if train:
            data, labels = self._make_imbalanced(data, labels, imbalance_factor)

        self.data   = data
        self.labels = labels

        # ── Class statistics (used by loss functions) ────────────────────────
        counts = np.bincount(self.labels, minlength=self.num_classes).astype(np.float32)
        self.class_freq      = torch.from_numpy(counts)                   # (K,) raw counts
        self.class_freq_norm = torch.from_numpy(counts / counts.sum())    # (K,) = pi_y

        # Shot partition thresholds  (Section 4.1)
        self.many_threshold   = 100   # > 100 → many-shot
        self.medium_threshold = 20    # 20–100 → medium-shot; < 20 → few-shot

    # ── Exponential imbalance sub-sampling ──────────────────────────────────

    def _make_imbalanced(
        self,
        data:   np.ndarray,
        labels: np.ndarray,
        gamma:  int,
    ):
        """
        Sub-sample each class c to:
            N_c = N_max * lambda^c
        where  lambda = (1/gamma)^{1/(K-1)}
        and    N_max  = samples in class 0 (most frequent).

        Paper reference: Section 4.1  "N_j = N × λ^j, where λ ∈ (0,1)"
        """
        K   = self.num_classes
        # lambda computed so that N_{K-1} / N_0 = 1/gamma
        lam = (1.0 / gamma) ** (1.0 / (K - 1))

        counts_full = [int(np.sum(labels == c)) for c in range(K)]
        N_max = max(counts_full)

        selected_data, selected_labels = [], []
        for c in range(K):
            n_target = max(1, int(N_max * (lam ** c)))
            idx_c    = np.where(labels == c)[0]
            np.random.shuffle(idx_c)
            idx_c    = idx_c[:n_target]
            selected_data.append(data[idx_c])
            selected_labels.append(np.full(len(idx_c), c, dtype=np.int64))

        return (
            np.concatenate(selected_data,   axis=0),
            np.concatenate(selected_labels, axis=0),
        )

    # ── Shot-partition helper ────────────────────────────────────────────────

    def get_shot_masks(self):
        """
        Return boolean masks for Many / Medium / Few-shot classes.
        Used during evaluation to compute per-group accuracy.
        Paper Section 4.1: Many>100, Medium 20-100, Few<20.
        """
        counts = self.class_freq.numpy()
        many   = counts >  self.many_threshold
        few    = counts <  self.medium_threshold
        medium = ~many & ~few
        return many, medium, few

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img   = Image.fromarray(self.data[idx])
        label = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label
