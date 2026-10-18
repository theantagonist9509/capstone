# %% [markdown]
# # Shapes (4-class) — Ensemble boundary traversal
#
# Loads **all seed classifiers** (`checkpoints/shapes/conv_classifier/seed_*/`,
# see train_classifier.py) plus one latent model (ConvAutoencoder, or ConvVAE
# with `--arch conv_vae` acting in mu-space) and, for each selected validation
# image, walks every classifier's latent space to the **same decision
# boundary**: the pair (source, target) = highest / second-highest mean
# probability mass of the **ensemble on the original image**. The walk stops
# when |P(source) − P(target)| ≤ EPSILON; overstepped steps are retried at a
# smaller lr (see traversal.extrapolate_image).
#
# Per image it saves a grid of the per-classifier boundary images, and pooled
# over all images it reports:
# 1. **Prob-mass balance** — histogram of the % difference
#    100·(p_target − p_source)/mean(p_source, p_target) at each boundary
#    image (should cluster at 0 if the walks land on the true boundary);
# 2. **Boundary-embedding variance** — mean_c ‖z_c − z̄‖² over the per-classifier
#    boundary latents (lower = the classifiers agree on where the boundary is).
#
# **Usage**
# - locally: `python runs/shapes/traverse.py [--smoke] [--arch conv_vae] [--vanilla]`
# - Kaggle:  `python kaggle.py runs/shapes/traverse.py` then
#   upload/push to the `btp-26` kernel (see `kaggle.md`)
#
# Requires trained checkpoints in `checkpoints/shapes/{conv_classifier/seed_*,conv_ae}`
# (or `conv_vae` with `--arch conv_vae`). `--classifier-seeds 42 47` restricts
# the ensemble to a subset of seeds.

# %%
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from custom_datasets import FolderDataset
from models import ConvAutoencoder, ConvClassifier, ConvVAE
from traversal import extrapolate_image, predict
from utils import load_best_model, maybe_show, print_checkpoint_info

# %% [markdown]
# ## Parameters

# %%
DATASET_DIR          = os.path.join(_repo_root, "dataset", "shapes")
CLS_CHECKPOINT_DIR   = os.path.join(_repo_root, "checkpoints", "shapes", "conv_classifier")
AE_CHECKPOINT_DIR    = os.path.join(_repo_root, "checkpoints", "shapes", "conv_ae")
VAE_CHECKPOINT_DIR   = os.path.join(_repo_root, "checkpoints", "shapes", "conv_vae")
OUT_DIR              = os.path.join(_repo_root, "runs", "shapes", "out")

IMAGE_SIZE           = 224
BASE_CHANNELS        = 32
N_PER_CLASS          = 2          # ensemble-correct images per class to traverse
STEPS                = 30         # max steps per boundary walk (early-stops at equality)
EPSILON              = 0.01       # stop when |P(source) − P(target)| ≤ EPSILON
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
class VAEAsAE(nn.Module):
    """
    Adapt a ConvVAE to the encoder/decoder interface traversal.py expects.

    encoder(x) returns the posterior mean mu — the deterministic latent
    point (eval-mode equivalent of sampling) — so every latent-space
    operation (steps, PCA bases, recon loop) acts in mu-space; decoder is
    the VAE's own pixel decoder (so `.parameters()` keeps working).
    ConvVAE deliberately has no `.encoder` attribute (see models.py) —
    this adapter is the only sanctioned bridge.
    """

    def __init__(self, vae):
        super().__init__()
        self._vae    = vae
        self.decoder = vae.decoder

    def encoder(self, x):
        mu, _ = self._vae.encode(x)
        return mu


def resolve_classifier_dirs(base_dir, seeds=None):
    """
    Classifiers are trained into `base_dir/seed_<seed>/` subdirs
    (see train_classifier.py). With seeds=None, use every seed dir in
    sorted order; a legacy flat layout (epoch_*.pth directly in base_dir)
    is still honoured as a single-dir ensemble.
    """
    if seeds:
        dirs = [os.path.join(base_dir, f"seed_{s}") for s in seeds]
        missing = [d for d in dirs if not os.path.isdir(d)]
        if missing:
            raise SystemExit(f"No classifier checkpoint dir(s): {missing} — "
                             f"run train_classifier.py first.")
        return dirs
    seed_dirs = sorted(
        d for d in glob.glob(os.path.join(base_dir, "seed_*")) if os.path.isdir(d)
    )
    if seed_dirs:
        return seed_dirs
    if glob.glob(os.path.join(base_dir, "epoch_*.pth")):
        return [base_dir]  # legacy flat layout
    raise SystemExit(
        f"No classifier checkpoints under '{base_dir}' — run train_classifier.py first."
    )


def load_classifier(dir_path, class_names, device):
    """Load the best-F1 checkpoint from one seed dir. Returns (label, model, ckpt)."""
    model = ConvClassifier(
        num_classes=len(class_names), base_channels=BASE_CHANNELS, image_size=IMAGE_SIZE
    ).to(device)
    ckpt = load_best_model(
        model, dir_path,
        index_selector=lambda ckpt: int(np.argmax(np.array(ckpt["history"]["val_macro_f1s"]))),
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = True
    return os.path.basename(dir_path), model, ckpt


def build_val_split():
    base = FolderDataset(root_dir=DATASET_DIR, transform=val_transform)
    n_total = len(base)
    n_val   = int(n_total * 0.2)
    n_train = n_total - n_val
    _, val_sub = torch.utils.data.random_split(
        base, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    return base.class_names, val_sub


def select_images(val_sub, class_names, n_per_class, classifier_models, device="cpu"):
    """Pick up to n_per_class val images per class whose ENSEMBLE prediction
    (mean softmax over all classifiers, on the original image) equals the true
    label. Misclassified images are skipped — we only traverse/show images the
    ensemble got right."""
    chosen = {i: [] for i in range(len(class_names))}
    scanned, rejected = 0, 0
    for idx in range(len(val_sub)):
        item = val_sub[idx]
        lbl_idx = item["label"].argmax().item()
        scanned += 1
        x = item["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            ens = torch.stack(
                [predict(clf, x)[1] for clf in classifier_models]
            ).mean(dim=0)
        if int(ens.argmax().item()) != lbl_idx:
            rejected += 1
            continue
        if len(chosen[lbl_idx]) < n_per_class:
            chosen[lbl_idx].append((item["image"], item["label"]))
        if all(len(v) == n_per_class for v in chosen.values()):
            break
    n_sel = sum(len(v) for v in chosen.values())
    print(f"Selected {n_sel} ensemble-correct images ({n_per_class}/class) "
          f"| scanned {scanned}, ensemble-misclassified skipped {rejected}")
    for i, name in enumerate(class_names):
        if not chosen[i]:
            print(f"WARNING: no ensemble-correct '{name}' images in the val split")
    return chosen


def plot_boundary_grid(original_img, rows, title, class_names,
                       source_idx, target_idx, save_path=None):
    """
    Original image + one boundary image per classifier.
    rows: [{"label", "image" (CHW), "p_source", "p_target", "pct_diff"}]
    """
    n = 1 + len(rows)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 3.2), squeeze=False)
    axes = axes[0]

    axes[0].imshow(
        (original_img.detach() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    )
    axes[0].set_title(f"original\n{class_names[source_idx]} → {class_names[target_idx]}",
                      fontsize=8)
    axes[0].axis("off")

    for ax, row in zip(axes[1:], rows):
        ax.imshow(
            (row["image"].detach() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        )
        ax.set_title(f"{row['label']}\np_s={row['p_source']:.2f} p_t={row['p_target']:.2f}\n"
                     f"Δ%={row['pct_diff']:+.1f}", fontsize=7)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot: {save_path}")
    maybe_show(fig)


# %% [markdown]
# ## Ensemble boundary demo

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="2 random classifiers, 1 image, 2 steps, no checkpoints")
    parser.add_argument("--arch", choices=["conv_ae", "conv_vae"], default="conv_ae",
                        help="which shape latent model to traverse "
                             "(conv_vae acts in mu-space; default: conv_ae)")
    parser.add_argument("--vanilla", action="store_true",
                        help="walk the raw score gradient instead of the "
                             "orthogonal (paper) traversal")
    parser.add_argument("--classifier-seeds", type=int, nargs="+", default=None,
                        metavar="SEED",
                        help="restrict the ensemble to these seed_<seed> dirs "
                             "(default: every seed dir found)")
    args = parser.parse_args()

    class_names, val_sub = build_val_split()
    print(f"Val split size: {len(val_sub)} | classes: {class_names}")
    print(f"Traversing arch: {args.arch}"
          + (" (VAE latent = mu, posterior mean)" if args.arch == "conv_vae" else "")
          + f" | mode: {'vanilla' if args.vanilla else 'orthogonal'}")

    # ── Latent model ──────────────────────────────────────────────────────────
    if args.arch == "conv_vae":
        raw_ae = ConvVAE(base_channels=BASE_CHANNELS, image_size=IMAGE_SIZE).to(DEVICE)
        ae_model = VAEAsAE(raw_ae)
    else:
        raw_ae = ConvAutoencoder(
            base_channels=BASE_CHANNELS, image_size=IMAGE_SIZE
        ).to(DEVICE)
        ae_model = raw_ae

    # Reconstruction-loop criterion — must match how the latent model was
    # trained (mse; see runs/shapes/train_{ae,vae}.py).
    rloss = nn.MSELoss()

    # ── Classifiers (ensemble) ────────────────────────────────────────────────
    if args.smoke:
        clf_labels = ["smoke0", "smoke1"]
        classifiers = [
            ConvClassifier(num_classes=len(class_names),
                           base_channels=BASE_CHANNELS, image_size=IMAGE_SIZE).to(DEVICE)
            for _ in range(2)
        ]
        for c in classifiers:
            c.eval()
    else:
        cls_dirs = resolve_classifier_dirs(CLS_CHECKPOINT_DIR, args.classifier_seeds)
        classifiers, clf_labels = [], []
        for d in cls_dirs:
            label, model, ckpt = load_classifier(d, class_names, DEVICE)
            print(f"Loaded classifier '{label}':")
            print_checkpoint_info(ckpt)
            clf_labels.append(label)
            classifiers.append(model)
        print(f"Ensemble size: {len(classifiers)} classifiers")

        ae_ckpt_dir = VAE_CHECKPOINT_DIR if args.arch == "conv_vae" else AE_CHECKPOINT_DIR
        ae_ckpt = load_best_model(
            raw_ae, ae_ckpt_dir,
            index_selector=(
                (lambda ckpt: int(np.argmin(np.array(ckpt["history"]["val_recons"]))))
                if args.arch == "conv_vae" else
                (lambda ckpt: int(np.argmin(np.array(ckpt["history"]["val_losses"]))))
            ),
            device=DEVICE,
        )
        print(f"{args.arch} checkpoint info:")
        print_checkpoint_info(ae_ckpt)

    ae_model.eval()
    for p in ae_model.decoder.parameters():
        p.requires_grad = True

    # ── Per-image: one boundary walk per classifier ───────────────────────────
    if args.smoke:
        chosen = {i: [] for i in range(len(class_names))}
        chosen[0] = [(val_sub[0]["image"], val_sub[0]["label"])]
    else:
        chosen = select_images(val_sub, class_names, N_PER_CLASS, classifiers, DEVICE)

    pct_diffs   = []            # pooled % prob-mass differences at boundaries
    var_rows    = []            # (label_name, k, latent_variance) per image
    n_images    = 0
    for lbl_idx, label_name in enumerate(class_names):
        if not chosen[lbl_idx]:
            print(f"Skipping class '{label_name}' — no ensemble-correct images.")
            continue
        for k, (img, lbl) in enumerate(chosen[lbl_idx]):
            n_images += 1
            x = img.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                ens_probs = torch.stack(
                    [predict(clf, x)[1] for clf in classifiers]
                ).mean(dim=0)
            ens_p = ens_probs.view(-1)
            src = int(ens_p.argmax().item())
            tgt = int(ens_p.argsort(descending=True)[1].item())
            print(f"\n=== {label_name} #{k+1}: ensemble source={class_names[src]} "
                  f"(p={float(ens_p[src]):.3f}) target={class_names[tgt]} "
                  f"(p={float(ens_p[tgt]):.3f}) ===")

            rows, z_list = [], []
            for label, clf in zip(clf_labels, classifiers):
                final_z = []
                results = extrapolate_image(
                    img.unsqueeze(0), lbl, clf, ae_model, rloss,
                    class_names=class_names, device=DEVICE,
                    steps=STEPS, orthogonal=not args.vanilla,
                    source=src, target=tgt, epsilon=EPSILON,
                    final_z_out=final_z,
                )
                boundary_img = results[0][-1]
                z_b = final_z[0]
                with torch.no_grad():
                    _, pb, _ = predict(clf, ae_model.decoder(z_b))
                pb = pb.view(-1)
                p_s, p_t = float(pb[src]), float(pb[tgt])
                pct = 100.0 * (p_t - p_s) / max(0.5 * (p_s + p_t), 1e-8)
                pct_diffs.append(pct)
                z_list.append(z_b.view(-1).float())
                rows.append({"label": label, "image": boundary_img,
                             "p_source": p_s, "p_target": p_t, "pct_diff": pct})
                print(f"  {label}: p_s={p_s:.4f} p_t={p_t:.4f} Δ%={pct:+.2f} "
                      f"({len(results[0]) - 1} steps)")

            plot_boundary_grid(
                img, rows,
                title=f"Boundary images — {label_name} #{k+1} "
                      f"({class_names[src]} = {class_names[tgt]})",
                class_names=class_names, source_idx=src, target_idx=tgt,
                save_path=os.path.join(OUT_DIR, f"boundary_{label_name}_{k+1}.png"),
            )

            # ── Metric 2: variance of the boundary embeddings ────────────────
            Z = torch.stack(z_list)                       # (C, D)
            z_bar = Z.mean(dim=0)
            var = float(((Z - z_bar) ** 2).sum(dim=1).mean())  # mean_c ‖z_c − z̄‖²
            var_rows.append((f"{label_name}#{k+1}", label_name, var))
            print(f"  boundary-embedding variance (mean ‖z_c − z̄‖²): {var:.4f}")

    if args.smoke:
        print(f"\n[SMOKE] {n_images} image × {len(classifiers)} classifiers ok | "
              f"{len(pct_diffs)} boundary walks | pooled Δ% sample: "
              f"{[round(p, 1) for p in pct_diffs[:4]]}")
        return

    # ── Metric 1: pooled histogram of prob-mass % difference ──────────────────
    if pct_diffs:
        arr = np.asarray(pct_diffs, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(arr, bins=30, alpha=0.8)
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.2, label="perfect balance")
        ax.axvline(arr.mean(), color="r", linestyle="--", linewidth=1.2,
                   label=f"mean {arr.mean():+.2f}%")
        ax.set_xlabel("100 · (p_target − p_source) / mean(p_source, p_target) at boundary")
        ax.set_ylabel("count")
        ax.set_title(f"Boundary prob-mass balance — {len(classifiers)} classifiers "
                     f"× {n_images} images")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        dist_path = os.path.join(OUT_DIR, "boundary_prob_balance.png")
        plt.savefig(dist_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot: {dist_path}")
        maybe_show(fig)
        print(f"[balance] Δ% across boundaries: mean {arr.mean():+.3f} ± {arr.std():.3f} "
              f"(min {arr.min():+.2f}, max {arr.max():+.2f}) | "
              f"|Δ%| ≤ 5: {(np.abs(arr) <= 5).mean() * 100:.0f}% of walks")
    else:
        print("[balance] no boundary walks collected — skipping histogram")

    # ── Metric 2 summary: per-image boundary-embedding variance ───────────────
    if var_rows:
        names  = [r[0] for r in var_rows]
        vars_  = [r[2] for r in var_rows]
        class_of = {r[0]: r[1] for r in var_rows}
        cmap = plt.get_cmap("tab10")
        colors = [cmap(class_names.index(class_of[n])) for n in names]
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(names)), 4.5))
        ax.bar(names, vars_, color=colors)
        ax.set_ylabel("mean ‖z_c − z̄‖² (lower = more agreement)")
        ax.set_title(f"Boundary-embedding variance per image — {len(classifiers)} classifiers")
        ax.grid(alpha=0.3, axis="y")
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(class_names))]
        ax.legend(handles, class_names, fontsize=8, title="image class")
        fig.tight_layout()
        var_path = os.path.join(OUT_DIR, "boundary_latent_variance.png")
        plt.savefig(var_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot: {var_path}")
        maybe_show(fig)
        print("[variance] per-image boundary-embedding variance:")
        for name, v in zip(names, vars_):
            print(f"  {name:14s} {v:.4f}")
        print(f"  mean across images: {np.mean(vars_):.4f} ± {np.std(vars_):.4f}")
    else:
        print("[variance] no variances collected — skipping plot")

    print("\nEnsemble boundary demo complete. Plots in:", OUT_DIR)


if __name__ == "__main__":
    main()
