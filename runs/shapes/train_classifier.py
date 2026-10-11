# ── Shapes (4-class) – ConvClassifier training ────────────────────────────────
#
# Trains the tiny ConvClassifier (models.ConvClassifier) on the 4-shapes
# dataset (circle / square / star / triangle) with softmax + cross-entropy.
#
# Usage:
#   python runs/shapes/train_classifier.py            # full training
#   python runs/shapes/train_classifier.py --smoke    # 1 fwd+bwd sanity pass
#
# Checkpoints: checkpoints/shapes/conv_classifier/epoch_*.pth

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from datasets import FolderDataset, TransformDataset

# ── Parameters ────────────────────────────────────────────────────────────────
DATASET_DIR      = os.path.join(_repo_root, "dataset", "shapes")
CHECKPOINT_DIR   = os.path.join(_repo_root, "checkpoints", "shapes", "conv_classifier")

IMAGE_SIZE       = 224
BASE_CHANNELS    = 32
DROPOUT          = 0.3
BATCH_SIZE       = 64
NUM_WORKERS      = 2
LEARNING_RATE    = 1e-3
NUM_EPOCHS       = 10
LABEL_SMOOTHING  = 0.1
VAL_SPLIT        = 0.2
SEED             = 42
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = 0.5
STD  = 0.5

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Using device  : {DEVICE}")
print(f"Checkpoint dir: {CHECKPOINT_DIR}")

# ── Transforms ────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([MEAN] * 3, [STD] * 3),
])

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([MEAN] * 3, [STD] * 3),
])


def build_loaders():
    base = FolderDataset(root_dir=DATASET_DIR, transform=None)
    n_total = len(base)
    n_val   = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val
    train_sub, val_sub = torch.utils.data.random_split(
        base, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    train_ds = TransformDataset(train_sub, train_transform)
    val_ds   = TransformDataset(val_sub, val_transform)

    # Class counts from the underlying one-hot labels (no image I/O)
    train_labels = [int(base.samples[i][1]) for i in train_sub.indices]
    class_counts = np.bincount(train_labels, minlength=len(base.class_names))
    print("Class counts (train):",
          {c: int(n) for c, n in zip(base.class_names, class_counts)})

    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=DEVICE == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=DEVICE == "cuda",
    )
    return base.class_names, train_loader, val_loader


def run_val(model, loader, criterion, device):
    model.eval()
    running_loss, total = 0.0, 0
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].argmax(dim=1).to(device, non_blocking=True)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            total        += imgs.size(0)
            all_logits.extend(logits.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    logits_arr = np.array(all_logits)
    targets_arr = np.array(all_targets)
    probs   = torch.softmax(torch.tensor(logits_arr), dim=1).numpy()
    preds   = logits_arr.argmax(axis=1)
    loss_val = running_loss / total
    acc  = accuracy_score(targets_arr, preds)
    f1   = f1_score(targets_arr, preds, average="macro", zero_division=0)
    auc  = roc_auc_score(targets_arr, probs, multi_class="ovr", average="macro")
    cm   = confusion_matrix(targets_arr, preds)
    return loss_val, acc, f1, auc, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="single fwd+bwd sanity pass, no checkpointing")
    args = parser.parse_args()

    class_names, train_loader, val_loader = build_loaders()

    from models import ConvClassifier
    model = ConvClassifier(
        num_classes=len(class_names),
        in_channels=3,
        base_channels=BASE_CHANNELS,
        image_size=IMAGE_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    if args.smoke:
        smoke_ds = Subset(train_loader.dataset, list(range(8)))
        imgs   = torch.stack([smoke_ds[i]["image"] for i in range(8)]).to(DEVICE)
        labels = torch.stack([smoke_ds[i]["label"] for i in range(8)]).argmax(dim=1).to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[SMOKE] images {tuple(imgs.shape)} → logits {tuple(logits.shape)} "
              f"→ loss {loss.item():.4f} | grads ok")
        return

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    history     = defaultdict(list)
    existing    = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pth")))
    if existing:
        ckpt = torch.load(existing[-1], map_location=DEVICE, weights_only=False)
        start_epoch = ckpt["epoch"] + 1
        history     = ckpt["history"]
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        print(f"Resumed from '{existing[-1]}' (epoch {ckpt['epoch']})")

    # ── Train ─────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        running_loss, total = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:>3}/{NUM_EPOCHS}] train", leave=False)
        for batch in pbar:
            imgs   = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].argmax(dim=1).to(DEVICE, non_blocking=True)
            logits = model(imgs)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            total        += imgs.size(0)
            pbar.set_postfix(loss=f"{running_loss / total:.4f}")

        train_loss = running_loss / total
        history["train_losses"].append(train_loss)

        val_loss, val_acc, val_f1, val_auc, val_cm = run_val(model, val_loader, criterion, DEVICE)
        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)
        history["val_macro_f1s"].append(val_f1)
        history["val_aucs"].append(val_auc)
        history["val_conf_matrices"].append(val_cm)

        scheduler.step(val_loss)
        print(
            f"Epoch [{epoch:>3}/{NUM_EPOCHS}]\n"
            f"  Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}\n"
            f"  Val Loss  : {val_loss:.4f} | Acc: {val_acc:.4f} | Macro F1: {val_f1:.4f} | AUC: {val_auc:.4f}\n"
            f"  Val CM:\n{val_cm}"
        )

        torch.save({
            "epoch"       : epoch,
            "history"     : dict(history),
            "model_state" : model.state_dict(),
            "optim_state" : optimizer.state_dict(),
            "sched_state" : scheduler.state_dict(),
        }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth"))

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
