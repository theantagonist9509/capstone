# %% [markdown]
# # Shapes (4-class) — ConvClassifier training
#
# Trains `NUM_CLASSIFIERS` independent ConvClassifier (`models.ConvClassifier`)
# runs on the 4-shapes dataset (circle / square / star / triangle) with softmax
# + cross-entropy — one per seed in `RUN_SEEDS` (`SEED .. SEED+N-1`). The
# train/val split is IDENTICAL across runs (fixed by `SEED`); each run differs
# only in weight init and sampler/augmentation order (per-run `torch.manual_seed`),
# so validation metrics are directly comparable across the ensemble — the seed
# spread is what the noise-vs-signal tests will use.
#
# **Usage**
# - locally: `python runs/shapes/train_classifier.py [--smoke]`
# - Kaggle:  `python kaggle.py runs/shapes/train_classifier.py` then
#   upload/push to the `btp-26` kernel (see `kaggle.md`)
#
# **Checkpoints** — `checkpoints/shapes/conv_classifier/seed_<seed>/epoch_*.pth`
# (one subdir per run; re-running resumes/skips per seed independently).
# Consumers (e.g. `runs/shapes/traverse.py --classifier-seed N`) load from one
# of these subdirs.

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
NUM_CLASSIFIERS  = 10         # independent runs; seeds SEED .. SEED+N-1
RUN_SEEDS        = [SEED + i for i in range(NUM_CLASSIFIERS)]
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS         = torch.cuda.device_count()
BATCH_SIZE       = BATCH_SIZE_PER_GPU * max(1, NUM_GPUS)

MEAN = 0.5
STD  = 0.5

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Using device  : {DEVICE} | {NUM_GPUS} GPU(s) | "
      f"batch {BATCH_SIZE} | workers {NUM_WORKERS}")
print(f"Checkpoint dir: {CHECKPOINT_DIR}")
print(f"Classifiers   : {NUM_CLASSIFIERS} runs | seeds {RUN_SEEDS}")

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
    # The split generator is seeded with the FIXED SEED (not the per-run seed),
    # so every classifier sees the same train/val partition — only the sampler
    # draw order / augmentations / init differ across runs (global torch RNG).
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
def make_model(class_names):
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
    return model


def train_one(run_seed, class_names):
    """Train (or resume) one classifier; checkpoints go to
    CHECKPOINT_DIR/seed_<run_seed>/. Returns the run's history dict."""
    run_dir = os.path.join(CHECKPOINT_DIR, f"seed_{run_seed}")
    os.makedirs(run_dir, exist_ok=True)

    # Fresh RNG for this run: different weight init + sampler/aug order.
    # (The data split itself is fixed — see build_loaders.)
    torch.manual_seed(run_seed)

    _, train_loader, val_loader = build_loaders()
    model = make_model(class_names)
    # Checkpoints always store the unwrapped weights (same layout on 1–2 GPUs).
    model_core = model.module if isinstance(model, nn.DataParallel) else model

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    history     = defaultdict(list)
    existing    = sorted(glob.glob(os.path.join(run_dir, "epoch_*.pth")))
    if existing:
        ckpt = torch.load(existing[-1], map_location=DEVICE, weights_only=False)
        if ckpt["epoch"] >= NUM_EPOCHS:
            print(f"[seed {run_seed}] already trained (epoch {ckpt['epoch']}) — "
                  f"reusing history, skipping training.")
            return ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        history     = ckpt["history"]
        model_core.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        print(f"[seed {run_seed}] resumed from '{existing[-1]}' (epoch {ckpt['epoch']})")

    # ── Train ─────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        running_loss, total = 0.0, 0
        pbar = tqdm(train_loader,
                    desc=f"seed {run_seed} | epoch [{epoch:>3}/{NUM_EPOCHS}] train",
                    leave=False)
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
            f"[seed {run_seed}] Epoch [{epoch:>3}/{NUM_EPOCHS}]\n"
            f"  Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}\n"
            f"  Val Loss  : {val_loss:.4f} | Acc: {val_acc:.4f} | "
            f"P: {val_p:.4f} | R: {val_r:.4f} | Macro F1: {val_f1:.4f} | AUC: {val_auc:.4f}\n"
            f"  Val CM:\n{val_cm}"
        )

        torch.save({
            "epoch"       : epoch,
            "seed"        : run_seed,
            "history"     : dict(history),
            "model_state" : model_core.state_dict(),
            "optim_state" : optimizer.state_dict(),
            "sched_state" : scheduler.state_dict(),
        }, os.path.join(run_dir, f"epoch_{epoch:03d}.pth"))

    return dict(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="single fwd+bwd sanity pass on one seed, no checkpointing")
    args = parser.parse_args()

    class_names, _, _ = build_loaders()

    if args.smoke:
        torch.manual_seed(RUN_SEEDS[0])
        _, train_loader, _ = build_loaders()
        model = make_model(class_names)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        smoke_ds = Subset(train_loader.dataset, list(range(8)))
        imgs   = torch.stack([smoke_ds[i]["image"] for i in range(8)]).to(DEVICE)
        labels = torch.stack([smoke_ds[i]["label"] for i in range(8)]).argmax(dim=1).to(DEVICE)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[SMOKE] seed {RUN_SEEDS[0]} | images {tuple(imgs.shape)} → "
              f"logits {tuple(logits.shape)} → loss {loss.item():.4f} | grads ok")
        return None

    # ── Train every seed (resumes/skips per seed independently) ───────────────
    histories = {}
    for i, run_seed in enumerate(RUN_SEEDS, start=1):
        print(f"\n{'=' * 70}\nClassifier run {i}/{NUM_CLASSIFIERS} — seed {run_seed}\n{'=' * 70}")
        histories[run_seed] = train_one(run_seed, class_names)

    # ── Ensemble summary: seed spread is the noise-vs-signal signal ───────────
    print(f"\n{'=' * 70}\nSummary across {len(histories)} classifiers\n{'=' * 70}")
    best_f1s, best_accs, best_aucs, final_f1s = [], [], [], []
    for run_seed, h in histories.items():
        best_epoch = int(np.argmax(np.array(h["val_macro_f1s"])))
        best_f1  = h["val_macro_f1s"][best_epoch]
        best_acc = h["val_accuracies"][best_epoch]
        best_auc = h["val_aucs"][best_epoch]
        best_f1s.append(best_f1); best_accs.append(best_acc); best_aucs.append(best_auc)
        final_f1s.append(h["val_macro_f1s"][-1])
        print(f"  seed {run_seed}: best-F1 {best_f1:.4f} @ epoch {best_epoch + 1} | "
              f"best acc {best_acc:.4f} | best AUC {best_auc:.4f} | "
              f"final F1 {h['val_macro_f1s'][-1]:.4f}")
    print(f"\nBest-F1 across seeds : mean {np.mean(best_f1s):.4f} ± {np.std(best_f1s):.4f} "
          f"(min {min(best_f1s):.4f}, max {max(best_f1s):.4f})")
    print(f"Best-acc across seeds: mean {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}")
    print(f"Best-AUC across seeds: mean {np.mean(best_aucs):.4f} ± {np.std(best_aucs):.4f}")
    print(f"Final-F1 across seeds: mean {np.mean(final_f1s):.4f} ± {np.std(final_f1s):.4f}")

    return histories


if __name__ == "__main__":
    histories = main()

# %% [markdown]
# ## Training curves & ensemble summary
#
# One curve per seed (thin), plus the cross-seed mean where lengths align.
# Rendered inline when running as a notebook (Kaggle); under a local script
# run this cell is skipped (`histories` is None).

# %%
if histories:
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(13, 4.5))
    for run_seed, h in histories.items():
        epochs = range(1, len(h["val_losses"]) + 1)
        ax_loss.plot(epochs, h["val_losses"], alpha=0.55, label=f"seed {run_seed}")
        ax_f1.plot(epochs, h["val_macro_f1s"], alpha=0.55, label=f"seed {run_seed}")

    # Cross-seed mean (only if every run trained the same number of epochs)
    lengths = {len(h["val_macro_f1s"]) for h in histories.values()}
    if len(lengths) == 1:
        epochs = range(1, next(iter(lengths)) + 1)
        mean_loss = np.mean([h["val_losses"] for h in histories.values()], axis=0)
        mean_f1   = np.mean([h["val_macro_f1s"] for h in histories.values()], axis=0)
        ax_loss.plot(epochs, mean_loss, "k-", linewidth=2, label="mean")
        ax_f1.plot(epochs, mean_f1, "k-", linewidth=2, label="mean")

    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("cross-entropy")
    ax_loss.set_title("Val loss per seed"); ax_loss.legend(fontsize=7); ax_loss.grid(alpha=0.3)
    ax_f1.set_xlabel("epoch"); ax_f1.set_ylabel("macro F1")
    ax_f1.set_title("Val macro-F1 per seed"); ax_f1.legend(fontsize=7); ax_f1.grid(alpha=0.3)
    fig.tight_layout()
    maybe_show(fig)

    # Final breakdown on each run's last epoch
    print("Final validation breakdown per seed (last epoch):")
    for run_seed, h in histories.items():
        print(f"  seed {run_seed}: acc {h['val_accuracies'][-1]:.4f} | "
              f"P {h['val_precisions'][-1]:.4f} | R {h['val_recalls'][-1]:.4f} | "
              f"F1 {h['val_macro_f1s'][-1]:.4f} | AUC {h['val_aucs'][-1]:.4f}")
    for key, label in [
        ("val_accuracies", "accuracy"),
        ("val_precisions", "precision (macro)"),
        ("val_recalls", "recall (macro)"),
        ("val_macro_f1s", "F1 (macro)"),
        ("val_aucs", "ROC-AUC (macro)"),
    ]:
        finals = [h[key][-1] for h in histories.values()]
        print(f"  {label:17s}: mean {np.mean(finals):.4f} ± {np.std(finals):.4f} "
              f"(min {min(finals):.4f}, max {max(finals):.4f})")
else:
    print("[curves] no training history in this session (smoke run?) — skipping plots")
