# %% [markdown]
# # Shapes (4-class) — ConvAutoencoder training
#
# Trains the tiny ConvAutoencoder (`models.ConvAutoencoder`) on the 4-shapes
# dataset. No labels are used; every image is a reconstruction target.
#
# **Usage**
# - locally: `python runs/shapes/train_ae.py [--smoke]`
# - Kaggle:  `python tools/py2nb.py runs/shapes/train_ae.py` then
#   `kaggle kernels push` (see `kaggle.md`)
#
# **Checkpoints** — `checkpoints/shapes/conv_ae/epoch_*.pth`

# %%
import argparse
import glob
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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
CHECKPOINT_DIR   = os.path.join(_repo_root, "checkpoints", "shapes", "conv_ae")

IMAGE_SIZE       = 224
BASE_CHANNELS    = 32
BATCH_SIZE_PER_GPU = 64
NUM_WORKERS      = min(4, (os.cpu_count() or 2) - 1)
LEARNING_RATE    = 1e-3
NUM_EPOCHS       = 10
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

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([MEAN] * 3, [STD] * 3),
])


def build_loaders():
    base = FolderDataset(root_dir=DATASET_DIR, transform=transform)
    n_total = len(base)
    n_val   = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val
    train_sub, val_sub = torch.utils.data.random_split(
        base, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(
        train_sub, batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS,
    )
    val_loader = DataLoader(
        val_sub, batch_size=BATCH_SIZE, shuffle=False, **LOADER_KWARGS,
    )
    return train_loader, val_loader


def run_val(model, loader, criterion, device):
    model.eval()
    running_loss, total = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            imgs  = batch["image"].to(device, non_blocking=True)
            recons = model(imgs)
            loss   = criterion(recons, imgs)
            running_loss += loss.item() * imgs.size(0)
            total        += imgs.size(0)
    return running_loss / total

# %% [markdown]
# ## Training

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="single fwd+bwd sanity pass, no checkpointing")
    args = parser.parse_args()

    train_loader, val_loader = build_loaders()

    from models import ConvAutoencoder
    model = ConvAutoencoder(
        in_channels=3,
        base_channels=BASE_CHANNELS,
        image_size=IMAGE_SIZE,
    ).to(DEVICE)
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
    # Checkpoints always store the unwrapped weights (same layout on 1–2 GPUs).
    model_core = model.module if isinstance(model, nn.DataParallel) else model

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if args.smoke:
        smoke_ds = Subset(train_loader.dataset, list(range(8)))
        imgs = torch.stack([smoke_ds[i]["image"] for i in range(8)]).to(DEVICE)
        recons = model(imgs)
        loss   = criterion(recons, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        z = model_core.encoder(imgs)
        print(f"[SMOKE] images {tuple(imgs.shape)} → recon {tuple(recons.shape)} "
              f"→ latent {tuple(z.shape)} → loss {loss.item():.4f} | grads ok")
        return None, model

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    history     = defaultdict(list)
    existing    = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pth")))
    if existing:
        ckpt = torch.load(existing[-1], map_location=DEVICE, weights_only=False)
        try:
            model_core.load_state_dict(ckpt["model_state"])
        except RuntimeError:
            # e.g. latent-size change: quarantine unusable old-arch checkpoints
            # so this run (and later traverse.py) don't trip over them.
            stale_dir = os.path.join(CHECKPOINT_DIR, "stale_old_arch")
            os.makedirs(stale_dir, exist_ok=True)
            for f in existing:
                os.replace(f, os.path.join(stale_dir, os.path.basename(f)))
            print(f"Architecture changed — moved {len(existing)} old checkpoint(s) "
                  f"to {stale_dir}; starting fresh.")
        else:
            start_epoch = ckpt["epoch"] + 1
            history     = ckpt["history"]
            optimizer.load_state_dict(ckpt["optim_state"])
            print(f"Resumed from '{existing[-1]}' (epoch {ckpt['epoch']})")

    # ── Train ─────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        running_loss, total = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:>3}/{NUM_EPOCHS}] train", leave=False)
        for batch in pbar:
            imgs   = batch["image"].to(DEVICE, non_blocking=True)
            recons = model(imgs)
            loss   = criterion(recons, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            total        += imgs.size(0)
            pbar.set_postfix(loss=f"{running_loss / total:.4f}")

        train_loss = running_loss / total
        val_loss   = run_val(model, val_loader, criterion, DEVICE)
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)

        print(
            f"Epoch [{epoch:>3}/{NUM_EPOCHS}]\n"
            f"  Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}\n"
            f"  Val Loss  : {val_loss:.4f}"
        )

        torch.save({
            "epoch"       : epoch,
            "history"     : dict(history),
            "model_state" : model_core.state_dict(),
            "optim_state" : optimizer.state_dict(),
        }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth"))

    print("\nTraining complete.")
    return dict(history), model


if __name__ == "__main__":
    history, model = main()

# %% [markdown]
# ## Training curves & reconstruction samples
#
# Left: MSE reconstruction loss on train/val. Right: a grid of val images
# (top) vs their reconstructions (bottom) from the final model.

# %%
if history is not None and history.get("val_losses"):
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = range(1, len(history["train_losses"]) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_losses"], label="train")
    ax.plot(epochs, history["val_losses"],   label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    ax.set_title("Autoencoder reconstruction loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    maybe_show(fig)

    print(f"Final losses — train: {history['train_losses'][-1]:.4f} | "
          f"val: {history['val_losses'][-1]:.4f}")

    # Reconstruction grid from the final in-memory model
    _, val_loader = build_loaders()
    batch = next(iter(val_loader))
    imgs = batch["image"][:8].to(DEVICE)
    with torch.no_grad():
        recons = model(imgs)

    imgs_np    = imgs.cpu().numpy()
    recons_np  = recons.cpu().numpy()
    n = imgs_np.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4.2))
    for i in range(n):
        axes[0, i].imshow(np.clip(imgs_np[i].transpose(1, 2, 0) * STD + MEAN, 0, 1))
        axes[1, i].imshow(np.clip(recons_np[i].transpose(1, 2, 0) * STD + MEAN, 0, 1))
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=10)
    axes[1, 0].set_ylabel("recon", fontsize=10)
    fig.suptitle("Validation reconstructions")
    fig.tight_layout()
    maybe_show(fig)
else:
    print("[curves] no training history in this session (smoke run?) — skipping plots")
