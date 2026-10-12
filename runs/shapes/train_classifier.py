# %% [markdown]
# # Shapes (4-class) — ConvClassifier training
#
# Trains the tiny ConvClassifier (`models.ConvClassifier`) on the 4-shapes
# dataset (circle / square / star / triangle) with softmax + cross-entropy.
#
# **Usage**
# - locally: `python runs/shapes/train_classifier.py [--smoke]`
# - Kaggle:  `python tools/py2nb.py runs/shapes/train_classifier.py` then
#   `kaggle kernels push` (see `kaggle.md`)
#
# **Checkpoints** — `checkpoints/shapes/conv_classifier/epoch_*.pth`

# %%
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
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from custom_datasets import FolderDataset, TransformDataset
from utils import maybe_show

# %% [markdown]
# ## Parameters

# %%
DATASET_DIR      = os.path.join(_repo_root, "dataset", "shapes")
CHECKPOINT_DIR   = os.path.join(_repo_root, "checkpoints", "shapes", "conv_classifier")

IMAGE_SIZE       = 224
BASE_CHANNELS    = 32
DROPOUT          = 0.3
BATCH_SIZE_PER_GPU = 64
NUM_WORKERS      = min(4, (os.cpu_count() or 2) - 1)
LEARNING_RATE    = 1e-3
NUM_EPOCHS       = 10
LABEL_SMOOTHING  = 0.1
VAL_SPLIT        = 0.2
SEED             = 42
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS         = torch.cuda.device_count()
BATCH_SIZE       = BATCH_SIZE_PER_GPU * max(1, NUM_GPUS)

MEAN = 0.5
STD  = 0.5

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Using device  : {DEVICE} | {NUM_GPUS} GPU(s) | "
      f"batch {BATCH_SIZE} | workers {NUM_WORKERS}")
print(f"Checkpoint dir: {CHECKPOINT_DIR}")

# %% [markdown]
# ## Data

# %%
LOADER_KWARGS = dict(
    num_workers=NUM_WORKERS,
    pin_memory=DEVICE == "cuda",
    **({"persistent_workers": True, "prefetch_factor": 4}
       if NUM_WORKERS > 0 else {}),
)

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
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, **LOADER_KWARGS,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, **LOADER_KWARGS,
    )
    return base.class_names, train_loader, val_loader

# %% [markdown]
# ## Validation

# %%
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

    logits_arr  = np.array(all_logits)
    targets_arr = np.array(all_targets)
    probs   = torch.softmax(torch.tensor(logits_arr), dim=1).numpy()
    preds   = logits_arr.argmax(axis=1)
    loss_val = running_loss / total
    acc  = accuracy_score(targets_arr, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets_arr, preds, average="macro", zero_division=0
    )
    auc  = roc_auc_score(targets_arr, probs, multi_class="ovr", average="macro")
    cm   = confusion_matrix(targets_arr, preds)
    return loss_val, acc, macro_p, macro_r, macro_f1, auc, cm

# %% [markdown]
# ## Training

# %%
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
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
    # Checkpoints always store the unwrapped weights (same layout on 1–2 GPUs).
    model_core = model.module if isinstance(model, nn.DataParallel) else model

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
        return None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    history     = defaultdict(list)
    existing    = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pth")))
    if existing:
        ckpt = torch.load(existing[-1], map_location=DEVICE, weights_only=False)
        start_epoch = ckpt["epoch"] + 1
        history     = ckpt["history"]
        model_core.load_state_dict(ckpt["model_state"])
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

        val_loss, val_acc, val_p, val_r, val_f1, val_auc, val_cm = run_val(
            model, val_loader, criterion, DEVICE
        )
        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)
        history["val_precisions"].append(val_p)
        history["val_recalls"].append(val_r)
        history["val_macro_f1s"].append(val_f1)
        history["val_aucs"].append(val_auc)
        history["val_conf_matrices"].append(val_cm)

        scheduler.step(val_loss)
        print(
            f"Epoch [{epoch:>3}/{NUM_EPOCHS}]\n"
            f"  Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}\n"
            f"  Val Loss  : {val_loss:.4f} | Acc: {val_acc:.4f} | "
            f"P: {val_p:.4f} | R: {val_r:.4f} | Macro F1: {val_f1:.4f} | AUC: {val_auc:.4f}\n"
            f"  Val CM:\n{val_cm}"
        )

        torch.save({
            "epoch"       : epoch,
            "history"     : dict(history),
            "model_state" : model_core.state_dict(),
            "optim_state" : optimizer.state_dict(),
            "sched_state" : scheduler.state_dict(),
        }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth"))

    print("\nTraining complete.")
    return dict(history)


if __name__ == "__main__":
    history = main()

# %% [markdown]
# ## Training curves & final metrics
#
# Rendered inline when running as a notebook (Kaggle); under a local script
# run this cell is skipped (`history` is None).

# %%
if history is not None and history.get("val_losses"):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_losses"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(epochs, history["train_losses"], label="train")
    ax_loss.plot(epochs, history["val_losses"],     label="val")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("cross-entropy")
    ax_loss.set_title("Loss"); ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["val_accuracies"], label="accuracy")
    ax_acc.plot(epochs, history["val_precisions"], label="precision (macro)")
    ax_acc.plot(epochs, history["val_recalls"],    label="recall (macro)")
    ax_acc.plot(epochs, history["val_macro_f1s"],  label="F1 (macro)")
    ax_acc.plot(epochs, history["val_aucs"],       label="ROC-AUC (macro)")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("score")
    ax_acc.set_title("Validation metrics"); ax_acc.legend(fontsize=8); ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    maybe_show(fig)

    # Final breakdown on the last epoch's predictions
    print("Final validation breakdown (last epoch):")
    print(f"  accuracy         : {history['val_accuracies'][-1]:.4f}")
    print(f"  precision (macro): {history['val_precisions'][-1]:.4f}")
    print(f"  recall    (macro): {history['val_recalls'][-1]:.4f}")
    print(f"  F1        (macro): {history['val_macro_f1s'][-1]:.4f}")
    print(f"  ROC-AUC   (macro): {history['val_aucs'][-1]:.4f}")
    print(f"\nConfusion matrix (last epoch):\n{history['val_conf_matrices'][-1]}")
else:
    print("[curves] no training history in this session (smoke run?) — skipping plots")
