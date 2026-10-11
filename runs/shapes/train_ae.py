# ── Shapes (4-class) – ConvAutoencoder training ───────────────────────────────
#
# Trains the tiny ConvAutoencoder (models.ConvAutoencoder) on the 4-shapes
# dataset. No labels are used; every image is a reconstruction target.
#
# Usage:
#   python runs/shapes/train_ae.py            # full training
#   python runs/shapes/train_ae.py --smoke    # 1 fwd+bwd sanity pass
#
# Checkpoints: checkpoints/shapes/conv_ae/epoch_*.pth

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

from datasets import FolderDataset, TransformDataset

# ── Parameters ────────────────────────────────────────────────────────────────
DATASET_DIR      = os.path.join(_repo_root, "dataset", "shapes")
CHECKPOINT_DIR   = os.path.join(_repo_root, "checkpoints", "shapes", "conv_ae")

IMAGE_SIZE       = 224
BASE_CHANNELS    = 32
BATCH_SIZE       = 64
NUM_WORKERS      = 2
LEARNING_RATE    = 1e-3
NUM_EPOCHS       = 10
VAL_SPLIT        = 0.2
SEED             = 42
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = 0.5
STD  = 0.5

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Using device  : {DEVICE}")
print(f"Checkpoint dir: {CHECKPOINT_DIR}")

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
        train_sub, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=DEVICE == "cuda",
    )
    val_loader = DataLoader(
        val_sub, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=DEVICE == "cuda",
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
        z = model.encoder(imgs)
        print(f"[SMOKE] images {tuple(imgs.shape)} → recon {tuple(recons.shape)} "
              f"→ latent {tuple(z.shape)} → loss {loss.item():.4f} | grads ok")
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
            "model_state" : model.state_dict(),
            "optim_state" : optimizer.state_dict(),
        }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth"))

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
