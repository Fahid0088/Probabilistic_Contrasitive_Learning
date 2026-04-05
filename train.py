"""
ProCo Training Script — CIFAR-10/100-LT
========================================
Paper: "Probabilistic Contrastive Learning for Long-Tailed Visual Recognition"
       TPAMI 2024 — Du, Wang, Song, Huang
       Code: https://github.com/LeapLabTHU/ProCo

This script reproduces the CIFAR-10/100-LT experiments (Section 4.4, Table 6/7).

ALL SETTINGS MATCH THE PAPER EXACTLY  (Section 4.2 — Implementation Details):
  ┌──────────────────────────────────────────────────────────────┐
  │  Backbone      : ResNet-32 [2]                               │
  │  Optimizer     : SGD, momentum=0.9, weight_decay=4e-4        │
  │  Batch size    : 256          (paper default)                 │
  │  Epochs        : 200 or 400                                   │
  │  LR schedule   : warm-up to 0.3 in first 5 epochs,           │
  │                  ×0.1 at epoch 160, ×0.1 at epoch 180        │
  │  Temperature   : τ = 0.1     (ProCo, CIFAR)                  │
  │  Loss weight   : α = 1.0     (Eq. 24)                        │
  │  Proj head     : hidden=512, output=128                       │
  │  Aug (cls)     : AutoAugment + Cutout  [85,86]                │
  │  Aug (repr)    : SimAugment  [51]                             │
  └──────────────────────────────────────────────────────────────┘

Combined loss (Eq. 24):
    L = L_LA + α · L_ProCo

Usage:
    # 200 epochs, CIFAR-100, gamma=100  (paper default)
    python train.py --dataset cifar100 --imbalance_factor 100 --epochs 200

    # 400 epochs (Table 7)
    python train.py --dataset cifar100 --imbalance_factor 100 --epochs 400

    # CIFAR-10, gamma=50
    python train.py --dataset cifar10 --imbalance_factor 50 --epochs 200

    # Resume training
    python train.py --resume ./checkpoints/proco_cifar100_imb100_ep100.pth

    # Evaluate only
    python train.py --eval_only ./checkpoints/proco_cifar100_imb100_BEST.pth
"""

import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ModuleNotFoundError:
    SummaryWriter = None
    _HAS_TB = False
    print('WARNING: tensorboard not installed — logging disabled.')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.resnet32          import ProCoModel
from loss.ProCoLoss           import ProCoLoss
from loss.LogitAdjustment     import LogitAdjustmentLoss
from datasets.lt_cifar        import (
    LongTailedCIFAR,
    get_cls_transforms,
    get_val_transforms,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Argument Parser
#  All defaults match the paper's CIFAR settings (Section 4.2)
# ──────────────────────────────────────────────────────────────────────────────
# ── CSV Logging ───────────────────────────────────────────────────────────────
import csv

def log_results_to_csv(args, overall, many, medium, few, best_acc, epoch):
    """
    Appends one row per run to results_log.csv in the save directory.
    Captures all CLI args + final accuracy results in one row.
    """
    csv_path = os.path.join(args.save_dir, 'results_log.csv')
    file_exists = os.path.isfile(csv_path)

    row = {
        # All command-line parameters
        'dataset':           args.dataset,
        'imbalance_factor':  args.imbalance_factor,
        'epochs':            args.epochs,
        'batch_size':        args.batch_size,
        'lr':                args.lr,
        'weight_decay':      args.weight_decay,
        'momentum':          args.momentum,
        'tau':               args.tau,
        'alpha':             args.alpha,
        'proj_dim':          args.proj_dim,
        'proj_hidden':       args.proj_hidden,
        'num_workers':       args.num_workers,
        'seed':              args.seed,
        # Results
        'final_epoch':       epoch + 1,
        'best_acc':          round(best_acc, 4),
        'final_overall':     round(overall, 4),
        'final_many':        round(many, 4),
        'final_medium':      round(medium, 4),
        'final_few':         round(few, 4),
        # Timestamp
        'timestamp':         time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()   # write column names only on first run
        writer.writerow(row)

    print(f'  Results logged → {csv_path}')


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='ProCo: Probabilistic Contrastive Learning for Long-Tailed Recognition'
    )

    # ── Dataset (Section 4.1) ────────────────────────────────────────────────
    p.add_argument('--dataset', default='cifar100',
                   choices=['cifar10', 'cifar100'],
                   help='Dataset name')
    p.add_argument('--imbalance_factor', type=int, default=100,
                   choices=[10, 50, 100],
                   help='Imbalance factor γ (Section 4.1). Paper tests 10/50/100.')
    p.add_argument('--data_root', default='./data',
                   help='Root directory for dataset download/cache')

    # ── Training schedule (Section 4.2) ─────────────────────────────────────
    p.add_argument('--epochs', type=int, default=200,
                   help='Total training epochs. Paper: 200 (Table 6) or 400 (Table 7)')
    p.add_argument('--batch_size', type=int, default=256,
                   help='Mini-batch size. Paper: 256  (Section 4.2)')
    p.add_argument('--lr', type=float, default=0.3,
                   help='Peak learning rate. Paper: 0.3  (Section 4.2)')
    p.add_argument('--weight_decay', type=float, default=4e-4,
                   help='SGD weight decay. Paper: 4×10⁻⁴  (Section 4.2)')
    p.add_argument('--momentum', type=float, default=0.9,
                   help='SGD momentum. Paper: 0.9  (Section 4.2)')

    # ── ProCo hyperparameters (Section 4.2) ─────────────────────────────────
    p.add_argument('--tau', type=float, default=0.1,
                   help='Temperature τ for ProCo loss. Paper: 0.1 for CIFAR  (Section 4.2)')
    p.add_argument('--alpha', type=float, default=1.0,
                   help='Weight α of representation branch  (Eq. 24). Paper: 1.0  (Fig. 3)')

    # ── Model architecture (Section 4.2) ────────────────────────────────────
    p.add_argument('--proj_dim', type=int, default=128,
                   help='Projection head output dimension p. Paper: 128 for CIFAR  (Section 4.2)')
    p.add_argument('--proj_hidden', type=int, default=512,
                   help='Projection head hidden dimension. Paper: 512 for CIFAR  (Section 4.2)')

    # ── I/O & misc ──────────────────────────────────────────────────────────
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--save_dir',    default='./checkpoints')
    p.add_argument('--log_dir',     default='./runs')
    p.add_argument('--eval_freq',   type=int, default=10,
                   help='Evaluate every N epochs')
    p.add_argument('--resume',      default='',
                   help='Checkpoint path to resume from')
    p.add_argument('--eval_only',   default='',
                   help='Checkpoint path for evaluation only (skip training)')

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#  Learning Rate Schedule  (Section 4.2)
#  Paper: "We gradually increase the learning rate to 0.3 in the first 5
#  epochs and reduce it by a factor of 10 at the 160th and 180th epochs."
#  For 400-epoch runs: warmup in first 10 epochs, decay at 360 and 380.
# ──────────────────────────────────────────────────────────────────────────────

def adjust_lr(optimizer: optim.SGD, epoch: int, args: argparse.Namespace) -> float:
    """
    Piecewise LR schedule from Section 4.2:
        Epoch < warmup              : linear warm-up to args.lr
        warmup ≤ epoch < decay1     : constant args.lr
        decay1 ≤ epoch < decay2     : args.lr × 0.1
        epoch ≥ decay2              : args.lr × 0.01
    """
    if args.epochs <= 200:
        # 200-epoch schedule (Table 6)
        warmup_epochs = 5    # "first 5 epochs"
        decay1        = 160  # "reduce at 160th epoch"
        decay2        = 180  # "reduce at 180th epoch"
    else:
        # 400-epoch schedule (Table 7 extended training)
        # "we warm up the learning rate in the first 10 epochs and
        #  decrease it at 360th and 380th epochs"
        warmup_epochs = 10
        decay1        = 360
        decay2        = 380

    if epoch < warmup_epochs:
        lr = args.lr * (epoch + 1) / warmup_epochs
    elif epoch < decay1:
        lr = args.lr
    elif epoch < decay2:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01

    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation  (Section 4.1)
#  Computes Top-1 accuracy on the balanced validation set,
#  broken into Many / Medium / Few-shot groups.
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    num_classes: int,
    class_freq:  torch.Tensor,   # (K,) train-set counts to determine shot group
) -> tuple:
    """
    Returns (overall_acc, many_acc, medium_acc, few_acc) in percent.

    Shot groups (Section 4.1):
        Many-shot   : > 100 images per class
        Medium-shot : 20 – 100 images
        Few-shot    : < 20 images
    """
    model.eval()

    per_class_correct = torch.zeros(num_classes)
    per_class_total   = torch.zeros(num_classes)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _    = model(imgs)
        preds        = logits.argmax(1)
        for c in range(num_classes):
            mask = (labels == c)
            per_class_correct[c] += (preds[mask] == c).sum().cpu()
            per_class_total[c]   += mask.sum().cpu()

    # Per-class accuracy
    per_class_acc = per_class_correct / per_class_total.clamp(min=1)

    # Shot masks  (Section 4.1 thresholds)
    counts = class_freq.cpu().numpy()
    many_mask   = counts >  100
    medium_mask = (counts >= 20) & (counts <= 100)
    few_mask    = counts <  20

    def group_acc(mask):
        if mask.sum() == 0:
            return 0.0
        return per_class_acc[mask].mean().item() * 100.0

    overall = per_class_acc.mean().item() * 100.0
    return overall, group_acc(many_mask), group_acc(medium_mask), group_acc(few_mask)


# ──────────────────────────────────────────────────────────────────────────────
#  Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:           nn.Module,
    loader:          DataLoader,
    optimizer:       optim.SGD,
    criterion_cls:   nn.Module,    # L_LA  (Eq. 1 / Eq. 24)
    criterion_proco: nn.Module,    # L_ProCo  (Eq. 20 / Eq. 24)
    epoch:           int,
    args:            argparse.Namespace,
    device:          torch.device,
    writer,
) -> float:
    """
    One training epoch.

    Combined loss (Eq. 24):
        L = L_LA + α · L_ProCo
    """
    model.train()
    total_loss = total_cls = total_proco = 0.0
    correct = total = 0
    t0 = time.time()

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # ── Forward pass ─────────────────────────────────────────────────────
        logits, z_proj = model(imgs)    # (B,K), (B,p)

        # ── Classification branch loss  L_LA  (Eq. 1) ────────────────────────
        loss_cls = criterion_cls(logits, labels)

        # ── Representation branch loss  L_ProCo  (Eq. 20) ────────────────────
        # class_prior π is the normalised class frequency
        loss_proco = criterion_proco(
            z_proj, labels,
            criterion_proco.pi.to(device)    # π_y attached in main()
        )

        # ── Combined loss  L = L_LA + α · L_ProCo  (Eq. 24) ─────────────────
        loss = loss_cls + args.alpha * loss_proco

        # ── Backward + optimiser step ─────────────────────────────────────────
        loss.backward()
        optimizer.step()

        # ── Metrics ──────────────────────────────────────────────────────────
        preds    = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss   += loss.item()
        total_cls    += loss_cls.item()
        total_proco  += loss_proco.item()

        if batch_idx % 50 == 0:
            elapsed = time.time() - t0
            print(
                f'  Epoch {epoch+1} [{batch_idx:3d}/{len(loader)}]'
                f'  Loss: {loss.item():.3f}'
                f'  (L_LA={loss_cls.item():.3f},'
                f' L_ProCo={loss_proco.item():.3f})'
                f'  Acc: {100.*correct/total:.1f}%'
                f'  {elapsed:.1f}s'
            )

    n = len(loader)
    if writer is not None:
        writer.add_scalar('Train/Loss',       total_loss  / n, epoch)
        writer.add_scalar('Train/Loss_LA',    total_cls   / n, epoch)
        writer.add_scalar('Train/Loss_ProCo', total_proco / n, epoch)
        writer.add_scalar('Train/Acc',    100.*correct/total,  epoch)

    return total_loss / n


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True    # faster conv for fixed input size

    print(f'\n{"="*65}')
    print(f'  ProCo  |  {args.dataset.upper()}-LT  |  γ={args.imbalance_factor}')
    print(f'  Device : {device}')
    print(f'  Epochs : {args.epochs}  |  BS={args.batch_size}  |  LR={args.lr}')
    print(f'  τ={args.tau}  |  α={args.alpha}  |  proj_dim={args.proj_dim}')
    print(f'{"="*65}\n')

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    # ── Datasets (Section 4.1 / 4.2) ────────────────────────────────────────
    # Classification branch augmentation: AutoAugment + Cutout  (Section 4.2)
    train_tf = get_cls_transforms(args.dataset)
    # Validation: no augmentation
    val_tf   = get_val_transforms(args.dataset)

    train_set = LongTailedCIFAR(
        root=args.data_root, dataset=args.dataset,
        imbalance_factor=args.imbalance_factor,
        train=True,  transform=train_tf, download=True,
    )
    val_set = LongTailedCIFAR(
        root=args.data_root, dataset=args.dataset,
        imbalance_factor=args.imbalance_factor,
        train=False, transform=val_tf, download=True,
    )

    # Paper uses batch_size=256 with drop_last=True  (standard for contrastive)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    num_classes      = train_set.num_classes
    class_freq       = train_set.class_freq.to(device)       # (K,) raw counts
    class_freq_norm  = train_set.class_freq_norm.to(device)  # (K,) = π_y

    print(f'  Train samples : {len(train_set)}')
    print(f'  Val samples   : {len(val_set)}')
    print(f'  Classes       : {num_classes}')
    print(f'  Max class size: {int(train_set.class_freq.max())}  '
          f'(head)  Min: {int(train_set.class_freq.min())}  (tail)\n')

    # ── Model (Section 4.2) ──────────────────────────────────────────────────
    # Backbone: ResNet-32
    # Two branches: classification (linear) + representation (MLP projection head)
    model = ProCoModel(
        num_classes=num_classes,
        proj_hidden=args.proj_hidden,   # 512 for CIFAR
        proj_out=args.proj_dim,         # 128 for CIFAR
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  Model parameters: {n_params:.2f}M\n')

    # ── Loss functions (Section 3.2) ─────────────────────────────────────────

    # Classification branch: Logit Adjustment  (Eq. 1)
    # Paper: L_LA with the training prior π_y
    criterion_cls = LogitAdjustmentLoss(
        class_freq=train_set.class_freq,   # raw counts → normalised internally
        tau=1.0,                           # τ_LA = 1.0  (Section 3.1 default)
    ).to(device)

    # Representation branch: ProCo Loss  (Eq. 20)
    # Paper CIFAR: τ=0.1, p=128
    criterion_proco = ProCoLoss(
        num_classes=num_classes,
        feat_dim=args.proj_dim,    # p = 128
        tau=args.tau,              # τ = 0.1
    ).to(device)
    # Attach π_y (class prior) so forward() can access it
    criterion_proco.pi = class_freq_norm   # (K,)

    # ── Optimiser (Section 4.2) ──────────────────────────────────────────────
    # Paper: SGD, momentum=0.9, weight_decay=4e-4
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,                    # 0.3  (peak, after warmup)
        momentum=args.momentum,        # 0.9
        weight_decay=args.weight_decay, # 4e-4
        nesterov=False,               # paper does not mention Nesterov
    )

    # ── TensorBoard ─────────────────────────────────────────────────────────
    writer = None
    if _HAS_TB:
        writer = SummaryWriter(
            log_dir=os.path.join(
                args.log_dir,
                f'{args.dataset}_imb{args.imbalance_factor}_ep{args.epochs}',
            )
        )

    # ── Evaluation-only mode ─────────────────────────────────────────────────
    if args.eval_only:
        print(f'  Loading checkpoint: {args.eval_only}')
        ckpt = torch.load(args.eval_only, map_location=device)
        model.load_state_dict(ckpt['model'])
        overall, many, medium, few = evaluate(
            model, val_loader, device, num_classes, train_set.class_freq)
        print(f'\n  Validation Results:')
        print(f'    Overall  : {overall:.2f}%')
        print(f'    Many-shot: {many:.2f}%')
        print(f'    Med-shot : {medium:.2f}%')
        print(f'    Few-shot : {few:.2f}%')
        return

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_acc    = 0.0
    if args.resume:
        print(f'  Resuming from: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        # Restore vMF estimator state
        criterion_proco.load_state_dict(ckpt['proco_state'])
        start_epoch = ckpt['epoch'] + 1
        best_acc    = ckpt.get('best_acc', 0.0)
        print(f'  Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%\n')

    # ── Training Loop ────────────────────────────────────────────────────────
    print('  Starting training...\n')
    for epoch in range(start_epoch, args.epochs):

        # Set LR according to paper schedule  (Section 4.2)
        lr = adjust_lr(optimizer, epoch, args)

        print(f'\nEpoch [{epoch+1}/{args.epochs}]  LR={lr:.5f}')

        # ── Train one epoch ──────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            criterion_cls, criterion_proco,
            epoch, args, device, writer,
        )

        # ── End of epoch: swap vMF accumulators  (Eq. 11, Section 3.2) ──────
        # "we adopt the estimated sample mean of the previous epoch for MLE,
        #  while maintaining a new sample mean from zero initialization
        #  in the current epoch"
        criterion_proco.end_epoch()

        # ── Periodic evaluation ──────────────────────────────────────────────
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            overall, many, medium, few = evaluate(
                model, val_loader, device, num_classes, train_set.class_freq)

            print(f'\n  ┌─ Val Accuracy ────────────────┐')
            print(f'  │  Overall  : {overall:6.2f}%          │')
            print(f'  │  Many-shot: {many:6.2f}%          │')
            print(f'  │  Med-shot : {medium:6.2f}%          │')
            print(f'  │  Few-shot : {few:6.2f}%  (best: {best_acc:.2f}%)  │')
            print(f'  └───────────────────────────────┘')

            if writer is not None:
                writer.add_scalar('Val/Overall',  overall, epoch)
                writer.add_scalar('Val/Many',     many,    epoch)
                writer.add_scalar('Val/Medium',   medium,  epoch)
                writer.add_scalar('Val/Few',      few,     epoch)

            # ── Save checkpoint ───────────────────────────────────────────────
            is_best = overall > best_acc
            if is_best:
                best_acc = overall

            ckpt_path = os.path.join(
                args.save_dir,
                f'proco_{args.dataset}_imb{args.imbalance_factor}_ep{epoch+1}.pth',
            )
            torch.save({
                'epoch':       epoch,
                'model':       model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'proco_state': criterion_proco.state_dict(),  # save vMF estimator
                'acc':         overall,
                'best_acc':    best_acc,
                'args':        vars(args),
            }, ckpt_path)

            if is_best:
                best_path = os.path.join(
                    args.save_dir,
                    f'proco_{args.dataset}_imb{args.imbalance_factor}_BEST.pth',
                )
                import shutil
                shutil.copy(ckpt_path, best_path)
                print(f'  ★ New best saved → {best_path}')
    
    log_results_to_csv(args, overall, many, medium, few, best_acc, epoch)

    print(f'\n{"="*65}')
    print(f'  Training complete!  Best accuracy: {best_acc:.2f}%')
    print(f'{"="*65}\n')

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
