# ProCo — Paper-Exact Reproduction
**Probabilistic Contrastive Learning for Long-Tailed Visual Recognition**
*TPAMI 2024 — Du, Wang, Song, Huang*
Official repo: https://github.com/LeapLabTHU/ProCo


## Installation

```bash
pip install torch torchvision scipy tensorboard
```

---

## Reproducing Paper Results

### Table 6 — CIFAR-100-LT (200 epochs)

```bash
# γ = 100  → expected: 52.8%
python train.py --dataset cifar100 --imbalance_factor 100 --epochs 200

# γ = 50   → expected: 57.1%
python train.py --dataset cifar100 --imbalance_factor 50  --epochs 200

# γ = 10   → expected: 65.5%
python train.py --dataset cifar100 --imbalance_factor 10  --epochs 200
```

### Table 6 — CIFAR-10-LT (200 epochs)

```bash
# γ = 100  → expected: 85.9%
python train.py --dataset cifar10 --imbalance_factor 100 --epochs 200

# γ = 50   → expected: 88.2%
python train.py --dataset cifar10 --imbalance_factor 50  --epochs 200

# γ = 10   → expected: 91.9%
python train.py --dataset cifar10 --imbalance_factor 10  --epochs 200
```

### Table 7 — CIFAR-100-LT γ=100 (400 epochs)
Expected: Many=70.1%, Med=53.4%, Few=36.4%, All=54.2%

```bash
python train.py --dataset cifar100 --imbalance_factor 100 --epochs 400
```

---

## All Hyperparameters (exact paper values)

```bash
python train.py \
  --dataset          cifar100 \   # or cifar10
  --imbalance_factor 100 \        # γ ∈ {10, 50, 100}
  --epochs           200 \        # 200 or 400
  --batch_size       256 \        # Paper Section 4.2
  --lr               0.3 \        # peak LR after warmup
  --weight_decay     4e-4 \       # Paper Section 4.2
  --momentum         0.9 \        # Paper Section 4.2
  --tau              0.1 \        # τ for ProCo (CIFAR)  Section 4.2
  --alpha            1.0 \        # α for Eq. 24         Fig. 3
  --proj_dim         128 \        # projection output p  Section 4.2
  --proj_hidden      512 \        # projection hidden    Section 4.2
  --eval_freq        10           # evaluate every 10 epochs
```

---

## Project Structure

```
ProCo/
├── train.py                  # Main training script (paper-exact settings)
├── README.md
│
├── models/
│   └── resnet32.py           # ResNet-32 backbone + 2-branch ProCo model
│                             # (Section 3.2 — Overall Objective)
│
├── loss/
│   ├── ProCoLoss.py          # ProCo Loss L_ProCo (Proposition 1, Eq. 20)
│   │                         # vMF distribution, MLE estimation, Miller recurrence
│   └── LogitAdjustment.py    # Logit Adjustment L_LA (Eq. 1) — classification branch
│
└── datasets/
    └── lt_cifar.py           # Long-tailed CIFAR-10/100 (Section 4.1)
                              # Exponential sub-sampling N_j = N * λ^j
```

---

## Key Paper Equations Implemented

| Equation | Location | What it is |
|---|---|---|
| Eq. 1 | `LogitAdjustment.py` | L_LA — logit adjustment loss |
| Eq. 4–6 | `ProCoLoss.py :: log_Cp()` | vMF pdf and normaliser C_p(κ) |
| Eq. 8 | `ProCoLoss.py :: estimate_vmf_params()` | MLE mean direction μ_y |
| Eq. 10 | `ProCoLoss.py :: estimate_vmf_params()` | MLE concentration κ̂_y |
| Eq. 11 | `ProCoLoss.py :: update_mean()` | Online sample mean accumulation |
| Eq. 13/20 | `ProCoLoss.py :: forward()` | Closed-form L_ProCo = L_in |
| Eq. 22–23 | `ProCoLoss.py :: bessel_miller()` | Miller backward recurrence |
| Eq. 24 | `train.py :: train_one_epoch()` | L = L_LA + α · L_ProCo |

---

## Resume / Evaluate

```bash
# Resume from checkpoint
python train.py --resume ./checkpoints/proco_cifar100_imb100_ep100.pth \
                --dataset cifar100 --imbalance_factor 100 --epochs 200

# Evaluate best checkpoint (prints Many/Med/Few breakdown)
python train.py --eval_only ./checkpoints/proco_cifar100_imb100_BEST.pth \
                --dataset cifar100 --imbalance_factor 100
```

---

## Using the Official Repository

The official code by the paper authors is at:
```
https://github.com/LeapLabTHU/ProCo
```

To get the official code:
```bash
git clone https://github.com/LeapLabTHU/ProCo.git
cd ProCo
pip install -r requirements.txt

# Run CIFAR-100-LT γ=100 (200 epochs) — Table 6
python train.py --dataset cifar100 --imbalance_factor 100 --epochs 200
```

The files in this project are a fully annotated reproduction that
maps every line of code to the specific paper equation or section.
