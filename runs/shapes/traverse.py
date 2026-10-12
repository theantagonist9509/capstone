# %% [markdown]
# # Shapes (4-class) — Latent space traversal demo
#
# Loads the best shapes ConvClassifier + ConvAutoencoder checkpoints and runs
# multiclass latent traversal (`traversal.py`):
# - untargeted gradient traversal (target = second-best class, the default)
# - one explicitly targeted traversal
# - a short PCA-axis walk (sideways, iso-confidence directions)
#
# **Usage**
# - locally: `python runs/shapes/traverse.py [--smoke]`
# - Kaggle:  `python tools/py2nb.py runs/shapes/traverse.py` then
#   `kaggle kernels push` (see `kaggle.md`)
#
# Requires trained checkpoints in `checkpoints/shapes/{conv_classifier,conv_ae}`.

# %%
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from custom_datasets import FolderDataset
from losses import MS_SSIMLoss
from models import ConvAutoencoder, ConvClassifier
from traversal import (
    extrapolate_image,
    plot_traversal,
    predict,
    take_latent_step,
    take_pca_step,
)
from utils import load_best_model, maybe_show, print_checkpoint_info

# %% [markdown]
# ## Parameters

# %%
DATASET_DIR          = os.path.join(_repo_root, "dataset", "shapes")
CLS_CHECKPOINT_DIR   = os.path.join(_repo_root, "checkpoints", "shapes", "conv_classifier")
AE_CHECKPOINT_DIR    = os.path.join(_repo_root, "checkpoints", "shapes", "conv_ae")
OUT_DIR              = os.path.join(_repo_root, "runs", "shapes", "out")

IMAGE_SIZE           = 224
BASE_CHANNELS        = 32
N_PER_CLASS          = 2          # images per class to traverse
STEPS                = 15
PCA_STEPS            = 5
PCA_EMBED_SUBSET     = 128        # val images used for the PCA bases
TARGET_PROB          = 0.75
SEED                 = 42
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = 0.5
STD  = 0.5

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([MEAN] * 3, [STD] * 3),
])

# %% [markdown]
# ## Helpers

# %%
def build_val_split():
    base = FolderDataset(root_dir=DATASET_DIR, transform=val_transform)
    n_total = len(base)
    n_val   = int(n_total * 0.2)
    n_train = n_total - n_val
    _, val_sub = torch.utils.data.random_split(
        base, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    return base.class_names, val_sub


def recon_loop_loss(device):
    return MS_SSIMLoss(
        channels=3,
        denorm_mean=torch.tensor([MEAN, MEAN, MEAN]),
        denorm_std=torch.tensor([STD, STD, STD]),
    ).to(device)


def save_image_grid(images, path, title=""):
    """Simple row of unnormalised images."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, images):
        disp = (img.detach() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        ax.imshow(disp)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved grid: {path}")
    maybe_show(fig)


def select_images(val_sub, class_names, n_per_class, classifier_model=None, device="cpu"):
    """Pick up to n_per_class val images per class that the classifier
    classifies CORRECTLY (prediction on the original image == true label).
    Misclassified images are skipped entirely — we only traverse/show images
    the classifier got right."""
    chosen = {i: [] for i in range(len(class_names))}
    scanned, rejected = 0, 0
    for idx in range(len(val_sub)):
        item = val_sub[idx]
        lbl_idx = item["label"].argmax().item()
        scanned += 1
        if classifier_model is not None:
            with torch.no_grad():
                _, _, pred_idx = predict(
                    classifier_model, item["image"].unsqueeze(0).to(device)
                )
            if pred_idx != lbl_idx:
                rejected += 1
                continue
        if len(chosen[lbl_idx]) < n_per_class:
            chosen[lbl_idx].append((item["image"], item["label"]))
        if all(len(v) == n_per_class for v in chosen.values()):
            break
    n_sel = sum(len(v) for v in chosen.values())
    print(f"Selected {n_sel} correctly-classified images ({n_per_class}/class) "
          f"| scanned {scanned}, misclassified skipped {rejected}")
    for i, name in enumerate(class_names):
        if not chosen[i]:
            print(f"WARNING: no correctly-classified '{name}' images in the val split")
    return chosen

# %% [markdown]
# ## Traversal demo

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="1 image, 2 steps, random-init models, no checkpoints")
    args = parser.parse_args()

    class_names, val_sub = build_val_split()
    print(f"Val split size: {len(val_sub)} | classes: {class_names}")

    classifier_model = ConvClassifier(
        num_classes=len(class_names), base_channels=BASE_CHANNELS, image_size=IMAGE_SIZE
    ).to(DEVICE)
    ae_model = ConvAutoencoder(
        base_channels=BASE_CHANNELS, image_size=IMAGE_SIZE
    ).to(DEVICE)

    rloss = recon_loop_loss(DEVICE)

    if args.smoke:
        item = val_sub[0]
        img, lbl = item["image"].unsqueeze(0), item["label"]
        results = extrapolate_image(
            img, lbl, classifier_model, ae_model, rloss,
            class_names=class_names, device=DEVICE,
            steps=2, orthogonal=True, target="auto",
        )
        imgs, probs, errs, gnorms, pred_idxs, target_idx = results
        assert len(imgs) == 3  # initial + 2 steps
        z = ae_model.encoder(img.to(DEVICE))
        with torch.no_grad():
            _, _, pred0 = predict(classifier_model, ae_model.decoder(z))
        # one manual multiclass latent step (SGD)
        z_step = z.detach().clone().requires_grad_(True)
        opt = torch.optim.SGD([z_step], lr=0.1)
        take_latent_step(
            z_step, classifier_model, ae_model, rloss, 1.0, opt,
            orthogonal=False, target_class=target_idx,
        )
        # PCA-step sanity on a fresh latent
        z_pca = z.detach().clone()
        emb = torch.cat([val_sub[i]["image"].unsqueeze(0) for i in range(8)], dim=0).to(DEVICE)
        with torch.no_grad():
            all_emb = ae_model.encoder(emb)
        z_pca = take_pca_step(z_pca, classifier_model, ae_model, all_emb, 0, 1.0, 0.1)
        print(f"[SMOKE] traversal ok | target={class_names[target_idx]} "
              f"pred0={class_names[pred0]} p_target={[round(p, 3) for p in probs]} "
              f"recon_err={errs[0]:.4f} |G|={gnorms[0]:.4f} | pca step ok")
        return

    # ── Load best checkpoints ─────────────────────────────────────────────────
    cls_ckpt = load_best_model(
        classifier_model, CLS_CHECKPOINT_DIR,
        index_selector=lambda ckpt: int(np.argmax(np.array(ckpt["history"]["val_macro_f1s"]))),
        device=DEVICE,
    )
    print("Classifier checkpoint info:")
    print_checkpoint_info(cls_ckpt)

    ae_ckpt = load_best_model(
        ae_model, AE_CHECKPOINT_DIR,
        index_selector=lambda ckpt: int(np.argmin(np.array(ckpt["history"]["val_losses"]))),
        device=DEVICE,
    )
    print("Autoencoder checkpoint info:")
    print_checkpoint_info(ae_ckpt)

    classifier_model.eval()
    ae_model.eval()
    for p in ae_model.decoder.parameters():
        p.requires_grad = True
    for p in classifier_model.parameters():
        p.requires_grad = True

    # ── Untargeted (second-best) gradient traversal per class ─────────────────
    # Only correctly-classified images are traversed (and thus shown).
    chosen = select_images(val_sub, class_names, N_PER_CLASS, classifier_model, DEVICE)
    for lbl_idx, label_name in enumerate(class_names):
        if not chosen[lbl_idx]:
            print(f"Skipping class '{label_name}' — no correctly-classified images.")
            continue
        for k, (img, lbl) in enumerate(chosen[lbl_idx]):
            results = extrapolate_image(
                img.unsqueeze(0), lbl, classifier_model, ae_model, rloss,
                class_names=class_names, device=DEVICE,
                steps=STEPS, orthogonal=True, target_prob=TARGET_PROB, target="auto",
            )
            imgs, probs, errs, gnorms, pred_idxs, target_idx = results
            plot_traversal(
                imgs, probs, errs, gnorms,
                pred_idxs=pred_idxs, class_names=class_names,
                title=f"Shapes traversal {label_name} #{k+1} → {class_names[target_idx]}",
                mean=[MEAN] * 3, std=[STD] * 3,
                save_path=os.path.join(OUT_DIR, f"traverse_{label_name}_{k+1}.png"),
            )

    # ── One explicitly targeted traversal (first correct image → class 0) ─────
    first_available = next((i for i in range(len(class_names)) if chosen[i]), None)
    if first_available is None:
        raise SystemExit("No correctly-classified images available to traverse.")
    img0, lbl0 = chosen[first_available][0]
    results = extrapolate_image(
        img0.unsqueeze(0), lbl0, classifier_model, ae_model, rloss,
        class_names=class_names, device=DEVICE,
        steps=STEPS, orthogonal=True, target_prob=TARGET_PROB, target=0,
    )
    imgs, probs, errs, gnorms, pred_idxs, target_idx = results
    plot_traversal(
        imgs, probs, errs, gnorms,
        pred_idxs=pred_idxs, class_names=class_names,
        title=f"Shapes targeted traversal → {class_names[0]}",
        mean=[MEAN] * 3, std=[STD] * 3,
        save_path=os.path.join(OUT_DIR, f"traverse_targeted_{class_names[0]}.png"),
    )

    # ── PCA-axis walk (sideways at ~constant confidence) ──────────────────────
    with torch.no_grad():
        emb = torch.stack(
            [val_sub[i]["image"] for i in range(min(PCA_EMBED_SUBSET, len(val_sub)))]
        ).to(DEVICE)
        all_embeddings = ae_model.encoder(emb)

        z0 = ae_model.encoder(img0.unsqueeze(0).to(DEVICE))
        _, probs0, pred0 = predict(classifier_model, ae_model.decoder(z0))

    frames = [ae_model.decoder(z0).cpu().squeeze(0)]
    z = z0
    for _ in range(PCA_STEPS):
        z = take_pca_step(z, classifier_model, ae_model, all_embeddings, 0, 1.0, 0.05)
        with torch.no_grad():
            frames.append(ae_model.decoder(z).cpu().squeeze(0))

    save_image_grid(
        frames,
        os.path.join(OUT_DIR, "pca_base0.png"),
        title=f"PCA base 0 walk from a {class_names[pred0]} "
              f"(p={probs0[pred0].item():.2f})",
    )

    print("\nTraversal demo complete. Plots in:", OUT_DIR)


if __name__ == "__main__":
    main()
