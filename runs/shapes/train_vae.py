# %% [markdown]
# # Shapes (4-class) — ConvVAE training
#
# Trains the tiny ConvVAE (`models.ConvVAE`) on the 4-shapes dataset. Same
# trunk/decoder geometry as ConvAutoencoder (latent (B, 512, 2, 2)), but the
# latent is a diagonal Gaussian q(z|x) = N(mu, exp(logvar)) sampled via the
# reparameterisation trick. No labels are used.
#
# **Loss** — MSE reconstruction + β·KL(q(z|x) ‖ N(0, I)); the KL term is the
# mechanism intended to smooth the latent space for traversal. Tune with
# `--beta` (default 1e-4); optional `--kl-warmup-epochs N` ramps β linearly
# over the first N epochs (0 = constant β).
#
# **Usage**
# - locally: `python runs/shapes/train_vae.py [--smoke] [--beta 1e-3]`
# - Kaggle:  `python kaggle.py runs/shapes/train_vae.py` then
#   upload/push to the `btp-26` kernel (see `kaggle.md`)
#
# **Checkpoints** — `checkpoints/shapes/conv_vae/epoch_*.pth`
#
# For later traversal: use `vae.decode(mu)` (deterministic); there is no
# `.encoder` attribute by design — see the ConvVAE docstring in models.py.

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

from custom_datasets import FolderDataset
from utils import maybe_show

# %% [markdown]
# ## Parameters

# %%
DATASET_DIR      = os.path.join(_repo_root, "dataset", "shapes")
CHECKPOINT_DIR   = os.path.join(_repo_root, "checkpoints", "shapes", "conv_vae")

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


def recon_and_kl(criterion, recon, imgs, mu, logvar):
    """MSE reconstruction + analytic KL of a diagonal Gaussian to N(0, I)
    (per-image sum over latent dims, then mean over the batch)."""
    recon_loss = criterion(recon, imgs)
    kld = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3))
    )
    return recon_loss, kld


def run_val(model, loader, criterion, device, beta):
    # eval() mode → reparameterize() returns mu, so val is deterministic.
    model.eval()
    tot_recon, tot_kl, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            recons, mu, logvar = model(imgs)
            recon_loss, kld = recon_and_kl(criterion, recons, imgs, mu, logvar)
            b = imgs.size(0)
            tot_recon += recon_loss.item() * b
            tot_kl    += kld.item() * b
            n         += b
    return tot_recon / n, tot_kl / n

# %% [markdown]
# ## Training

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="single fwd+bwd sanity pass, no checkpointing")
    parser.add_argument("--beta", type=float, default=1e-4,
                        help="weight on the KL term (default: 1e-4)")
    parser.add_argument("--kl-warmup-epochs", type=int, default=0,
                        help="ramp β linearly over the first N epochs (0 = constant)")
    args = parser.parse_args()

    train_loader, val_loader = build_loaders()

    from models import ConvVAE
    model = ConvVAE(
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
        recons, mu, logvar = model(imgs)
        recon_loss, kld = recon_and_kl(criterion, recons, imgs, mu, logvar)
        loss = recon_loss + args.beta * kld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[SMOKE] images {tuple(imgs.shape)} → recon {tuple(recons.shape)} "
              f"→ mu {tuple(mu.shape)} (latent {mu[0].numel()}) → "
              f"recon {recon_loss.item():.4f} + {args.beta:g}·KL {kld.item():.2f} "
              f"= {loss.item():.4f} | grads ok")
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
            stale_dir = os.path.join(CHECKPOINT_DIR, "stale_old_arch")
            os.makedirs(stale_dir, exist_ok=True)
            for f in existing:
                os.replace(f, os.path.join(stale_dir, os.path.basename(f)))
            print(f"Checkpoint architecture changed — moved {len(existing)} "
                  f"old checkpoint(s) to {stale_dir}; starting fresh.")
        else:
            start_epoch = ckpt["epoch"] + 1
            history     = ckpt["history"]
            optimizer.load_state_dict(ckpt["optim_state"])
            prev_beta = ckpt.get("beta", args.beta)
            if prev_beta != args.beta:
                print(f"WARNING: resuming with --beta {args.beta:g} but history "
                      f"was trained with beta {prev_beta:g} — curves not comparable.")
            print(f"Resumed from '{existing[-1]}' (epoch {ckpt['epoch']})")

    # ── Train ─────────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        beta_t = (args.beta * min(1.0, epoch / args.kl_warmup_epochs)
                  if args.kl_warmup_epochs > 0 else args.beta)
        model.train()
        tot_recon, tot_kl, n = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:>3}/{NUM_EPOCHS}] train", leave=False)
        for batch in pbar:
            imgs   = batch["image"].to(DEVICE, non_blocking=True)
            recons, mu, logvar = model(imgs)
            recon_loss, kld = recon_and_kl(criterion, recons, imgs, mu, logvar)
            loss = recon_loss + beta_t * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            b = imgs.size(0)
            tot_recon += recon_loss.item() * b
            tot_kl    += kld.item() * b
            n         += b
            pbar.set_postfix(recon=f"{tot_recon / n:.4f}", kl=f"{tot_kl / n:.2f}")

        train_recon, train_kl = tot_recon / n, tot_kl / n
        val_recon, val_kl = run_val(model, val_loader, criterion, DEVICE, args.beta)
        val_loss = val_recon + args.beta * val_kl

        history["train_recons"].append(train_recon)
        history["train_kls"].append(train_kl)
        history["val_recons"].append(val_recon)
        history["val_kls"].append(val_kl)
        history["val_losses"].append(val_loss)

        print(
            f"Epoch [{epoch:>3}/{NUM_EPOCHS}]\n"
            f"  Train: recon {train_recon:.4f} | KL {train_kl:.2f} | "
            f"beta_t {beta_t:.2e} | LR {optimizer.param_groups[0]['lr']:.2e}\n"
            f"  Val  : recon {val_recon:.4f} | KL {val_kl:.2f} | "
            f"total {val_loss:.4f}"
        )

        torch.save({
            "epoch"       : epoch,
            "history"     : dict(history),
            "model_state" : model_core.state_dict(),
            "optim_state" : optimizer.state_dict(),
            "beta"        : args.beta,
        }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth"))

    print("\nTraining complete.")
    return dict(history), model


if __name__ == "__main__":
    history, model = main()

# %% [markdown]
# ## Training curves & reconstruction samples
#
# Left: MSE reconstruction (train/val). Middle: KL divergence (train/val).
# Right: a grid of val images (top) vs `decode(mu)` reconstructions (bottom)
# from the final model (eval mode — no sampling).

# %%
if history is not None and history.get("val_recons"):
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = range(1, len(history["val_recons"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_recons"], label="train")
    axes[0].plot(epochs, history["val_recons"],   label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE")
    axes[0].set_title("Reconstruction loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_kls"], label="train")
    axes[1].plot(epochs, history["val_kls"],   label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("KL (per image)")
    axes[1].set_title("KL divergence"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    maybe_show(fig)

    print(f"Final — recon train/val: {history['train_recons'][-1]:.4f} / "
          f"{history['val_recons'][-1]:.4f} | KL train/val: "
          f"{history['train_kls'][-1]:.2f} / {history['val_kls'][-1]:.2f}")

    # Reconstruction grid from the final in-memory model (deterministic mu)
    model.eval()
    _, val_loader = build_loaders()
    batch = next(iter(val_loader))
    imgs = batch["image"][:8].to(DEVICE)
    with torch.no_grad():
        recons, _, _ = model(imgs)

    imgs_np   = imgs.cpu().numpy()
    recons_np = recons.cpu().numpy()
    n = imgs_np.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4.2))
    for i in range(n):
        axes[0, i].imshow(np.clip(imgs_np[i].transpose(1, 2, 0) * STD + MEAN, 0, 1))
        axes[1, i].imshow(np.clip(recons_np[i].transpose(1, 2, 0) * STD + MEAN, 0, 1))
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=10)
    axes[1, 0].set_ylabel("recon (mu)", fontsize=10)
    fig.suptitle("Validation reconstructions — decode(mu)")
    fig.tight_layout()
    maybe_show(fig)
else:
    print("[curves] no training history in this session (smoke run?) — skipping plots")
