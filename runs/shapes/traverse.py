# %% [markdown]
# # Shapes (4-class) — Ensemble boundary traversal
#
# Loads **all seed classifiers** (`checkpoints/shapes/conv_classifier/seed_*/`,
# see train_classifier.py) plus one latent model (ConvAutoencoder, or ConvVAE
# with `--arch conv_vae` acting in mu-space) and, for each selected validation
# image, walks every classifier's latent space AND its pixel space (FGSM-like
# control) to the **same decision boundary**: the pair (source, target) =
# highest / second-highest mean probability mass of the **ensemble on the
# original image**. Walks stop when |P(source) − P(target)| ≤ EPSILON;
# overstepped steps are retried at a smaller lr (see traversal.py).
#
# Per image it saves a grid of the per-classifier boundary images, and pooled
# over all images it reports:
# 1. **Cross-classifier transfer** — every classifier is evaluated on every
#    boundary image (n×n matrix per walk type). The diagonal is the walk's
#    own convergence residual (mechanics sanity check); the OFF-diagonal is
#    the stability test: do independently-seeded classifiers also see
#    p_source ≈ p_target at another classifier's boundary? Shown as Δ%
#    = 100·(p_t − p_s)/mean(p_s, p_t) and as the absolute margin p_t − p_s
#    (guards against small-denominator artifacts), for latent and pixel
#    walks side by side. Softer companion: the **margin shift**
#    Δmargin_j = margin_j(boundary_i) − margin_j(walk start), where the walk
#    start is the AE reconstruction for latent walks (the classifier only
#    ever sees decoded images) and the original image for pixel walks —
#    does walking classifier i's boundary move classifier j toward the
#    target at all (shared direction), even when it doesn't reach equality?
# 2. **Boundary dispersion** — mean_c ‖z_c − z̄‖² over the per-classifier
#    boundary latents, normalised by the encoder's own spread across a val
#    subset (dimensionless ratio; lower = classifiers agree more). The
#    pixel-space boundary dispersion over the same normalisation is the
#    control (no shared latent model).
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
from traversal import extrapolate_image, extrapolate_pixels, predict
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
STEPS                = 30         # max steps per walk (early-stops at equality)
EPSILON              = 0.01       # stop when |P(source) − P(target)| ≤ EPSILON
REF_SUBSET           = 128        # val images for the encoder/pixel spread reference
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
    operation acts in mu-space; decoder is the VAE's own pixel decoder
    (so `.parameters()` keeps working). ConvVAE deliberately has no
    `.encoder` attribute (see models.py) — this adapter is the only
    sanctioned bridge.
    """

    def __init__(self, vae):
        super().__init__()
        self._vae    = vae
        self.decoder = vae.decoder

    def encoder(self, x):
        mu, _ = self._vae.encode(x)
        return mu


def _pct(p_t: float, p_s: float) -> float:
    """Symmetric percent difference of the two probs (guarding the denominator)."""
    return 100.0 * (p_t - p_s) / max(0.5 * (p_s + p_t), 1e-8)


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


def cross_eval(frames_x, classifiers, src, tgt):
    """Evaluate every classifier on every boundary frame.

    frames_x: list of C model-input tensors (1,3,H,W) on DEVICE.
    Returns (M, A): C×C matrices of symmetric % difference and of the
    absolute margin p_target − p_source.
    """
    C = len(classifiers)
    M = np.zeros((C, C))
    A = np.zeros((C, C))
    with torch.no_grad():
        for i in range(C):
            for j, clf_j in enumerate(classifiers):
                _, pb, _ = predict(clf_j, frames_x[i])
                pb = pb.view(-1)
                p_s, p_t = float(pb[src]), float(pb[tgt])
                A[i, j] = p_t - p_s
                M[i, j] = _pct(p_t, p_s)
    return M, A


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
          + f" | mode: {'vanilla' if args.vanilla else 'orthogonal'}"
          + f" | ε = {EPSILON:g}")

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

    # ── Spread references for the normalised dispersion metric ────────────────
    with torch.no_grad():
        emb = torch.stack(
            [val_sub[i]["image"] for i in range(min(REF_SUBSET, len(val_sub)))]
        ).to(DEVICE)
        Zd = ae_model.encoder(emb).view(emb.size(0), -1).float()
        latent_ref_var = float(((Zd - Zd.mean(dim=0)) ** 2).sum(dim=1).mean())
        Xd = emb.view(emb.size(0), -1).float()
        pixel_ref_var = float(((Xd - Xd.mean(dim=0)) ** 2).sum(dim=1).mean())
    print(f"[ref] spread over {min(REF_SUBSET, len(val_sub))} val images — "
          f"encoder (latent): {latent_ref_var:.2f} | pixel: {pixel_ref_var:.2f}")

    # ── Per-image: one latent walk + one pixel walk per classifier ────────────
    if args.smoke:
        chosen = {i: [] for i in range(len(class_names))}
        chosen[0] = [(val_sub[0]["image"], val_sub[0]["label"])]
    else:
        chosen = select_images(val_sub, class_names, N_PER_CLASS, classifiers, DEVICE)

    pools = {k: [] for k in ("lat_diag", "lat_off", "pix_diag", "pix_off",
                             "lat_off_abs", "pix_off_abs",
                             "lat_shift", "pix_shift", "lat_offset",
                             "lat_S", "pix_S")}
    first_shift = None   # (S_lat, S_pix, src, tgt, title) of the first image
    var_rows = []   # (display_name, class_name, lat_ratio, pix_ratio,
                    #  lat_var, lat_var_per_dim, pix_var)
    n_images = 0
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

            lat_zs, lat_xs, pix_xs = [], [], []
            lat_own, pix_own = [], []
            for label, clf in zip(clf_labels, classifiers):
                # — latent walk —
                final_z = []
                lat_s_hist = []
                results = extrapolate_image(
                    img.unsqueeze(0), lbl, clf, ae_model, rloss,
                    class_names=class_names, device=DEVICE,
                    steps=STEPS, orthogonal=not args.vanilla,
                    source=src, target=tgt, epsilon=EPSILON,
                    final_z_out=final_z,
                    pair_mass_out=lat_s_hist,
                )
                z_b   = final_z[0]
                xb    = ae_model.decoder(z_b)          # (1,3,H,W) on DEVICE
                _, pb, _ = predict(clf, xb)
                pb = pb.view(-1)
                p_s, p_t = float(pb[src]), float(pb[tgt])
                lat_zs.append(z_b)
                lat_xs.append(xb)
                lat_own.append((p_s, p_t, _pct(p_t, p_s)))
                print(f"  [latent] {label}: p_s={p_s:.4f} p_t={p_t:.4f} "
                      f"Δ%={_pct(p_t, p_s):+.2f} ({len(results[0]) - 1} steps) "
                      f"S={lat_s_hist[-1]:.3f} (start {lat_s_hist[0]:.3f})")

                # — pixel walk (control) —
                pix_s_hist = []
                r_pix = extrapolate_pixels(
                    img.unsqueeze(0), lbl, clf,
                    class_names=class_names, device=DEVICE,
                    steps=STEPS, source=src, target=tgt, epsilon=EPSILON,
                    mean=[MEAN] * 3, std=[STD] * 3,
                    pair_mass_out=pix_s_hist,
                )
                xi = r_pix[0][-1].unsqueeze(0).to(DEVICE)   # (1,3,H,W)
                _, pi, _ = predict(clf, xi)
                pi = pi.view(-1)
                ps_i, pt_i = float(pi[src]), float(pi[tgt])
                pix_xs.append(xi)
                pix_own.append((ps_i, pt_i, _pct(pt_i, ps_i)))
                print(f"  [pixels] {label}: p_s={ps_i:.4f} p_t={pt_i:.4f} "
                      f"Δ%={_pct(pt_i, ps_i):+.2f} ({len(r_pix[0]) - 1} steps) "
                      f"S={pix_s_hist[-1]:.3f} (start {pix_s_hist[0]:.3f})")
                pools["lat_S"].append(lat_s_hist[-1])
                pools["pix_S"].append(pix_s_hist[-1])

            # — cross-classifier transfer matrices —
            C = len(classifiers)
            M_lat, A_lat = cross_eval(lat_xs, classifiers, src, tgt)
            M_pix, A_pix = cross_eval(pix_xs, classifiers, src, tgt)
            off = ~np.eye(C, dtype=bool)
            pools["lat_diag"].extend(np.diag(M_lat))
            pools["pix_diag"].extend(np.diag(M_pix))
            pools["lat_off"].extend(M_lat[off])
            pools["pix_off"].extend(M_pix[off])
            pools["lat_off_abs"].extend(A_lat[off])
            pools["pix_off_abs"].extend(A_pix[off])

            # — margin shift vs each classifier's view of the WALK START —
            # The latent walk optimises z from encoder(x), but the classifier
            # only ever sees decoder(z), so its step-0 probabilities are those
            # of the reconstruction — that is the correct baseline. The pixel
            # walk starts at x itself, so its baseline is the original image.
            with torch.no_grad():
                x_recon = ae_model.decoder(ae_model.encoder(x))
            margins_recon = np.zeros(C)
            margins_orig = np.zeros(C)
            with torch.no_grad():
                for j, clf_j in enumerate(classifiers):
                    _, pr, _ = predict(clf_j, x_recon)
                    pr = pr.view(-1)
                    margins_recon[j] = float(pr[tgt]) - float(pr[src])
                    _, po, _ = predict(clf_j, x)
                    po = po.view(-1)
                    margins_orig[j] = float(po[tgt]) - float(po[src])
            S_lat = A_lat - margins_recon[None, :]   # what the walk changed
            S_pix = A_pix - margins_orig[None, :]    # pixel walk starts at x
            pools["lat_shift"].extend(S_lat[off])
            pools["pix_shift"].extend(S_pix[off])
            pools["lat_offset"].extend(margins_recon - margins_orig)
            if first_shift is None:
                first_shift = (S_lat.copy(), S_pix.copy(), src, tgt,
                               f"{label_name}#{k+1}")

            print(f"  transfer off-diag Δ%: latent {M_lat[off].mean():+.2f} ± "
                  f"{M_lat[off].std():.2f} | pixels {M_pix[off].mean():+.2f} ± "
                  f"{M_pix[off].std():.2f} | abs margin: latent "
                  f"{A_lat[off].mean():+.4f} | pixels {A_pix[off].mean():+.4f}")
            print(f"  transfer Δmargin (vs walk start): latent {S_lat[off].mean():+.4f} ± "
                  f"{S_lat[off].std():.4f} (toward target: "
                  f"{(S_lat[off] > 0).mean() * 100:.0f}%) | pixels "
                  f"{S_pix[off].mean():+.4f} ± {S_pix[off].std():.4f} "
                  f"(toward target: {(S_pix[off] > 0).mean() * 100:.0f}%)")

            if not args.smoke:
                rows = [
                    {"label": lab, "image": lat_xs[i].squeeze(0).cpu(),
                     "p_source": lat_own[i][0], "p_target": lat_own[i][1],
                     "pct_diff": lat_own[i][2]}
                    for i, lab in enumerate(clf_labels)
                ]
                plot_boundary_grid(
                    img, rows,
                    title=f"Boundary images (latent) — {label_name} #{k+1} "
                          f"({class_names[src]} → {class_names[tgt]}, ε = {EPSILON:g})",
                    class_names=class_names, source_idx=src, target_idx=tgt,
                    save_path=os.path.join(OUT_DIR, f"boundary_{label_name}_{k+1}.png"),
                )

            # ── Metric 2: boundary dispersion, normalised + pixel control ─────
            Z = torch.stack([z.view(-1) for z in lat_zs]).float()   # (C, D)
            lat_var = float(((Z - Z.mean(dim=0)) ** 2).sum(dim=1).mean())
            lat_var_per_dim = lat_var / Z.shape[1]
            lat_ratio = lat_var / max(latent_ref_var, 1e-12)
            Xp = torch.stack([t.view(-1) for t in pix_xs]).float()  # (C, 3HW)
            pix_var = float(((Xp - Xp.mean(dim=0)) ** 2).sum(dim=1).mean())
            pix_ratio = pix_var / max(pixel_ref_var, 1e-12)
            var_rows.append((f"{label_name}#{k+1}", label_name,
                             lat_ratio, pix_ratio, lat_var, lat_var_per_dim, pix_var))
            print(f"  dispersion: latent var {lat_var:.1f} "
                  f"(per-dim {lat_var_per_dim:.4f}, ratio {lat_ratio:.4f}) | "
                  f"pixel var {pix_var:.1f} (ratio {pix_ratio:.4f})")

    if args.smoke:
        n_lat = len(pools["lat_diag"]) + len(pools["lat_off"])
        n_sh  = len(pools["lat_shift"]) + len(pools["pix_shift"])
        print(f"\n[SMOKE] {n_images} image × {len(classifiers)} classifiers ok | "
              f"transfer entries: latent {n_lat}, pixels "
              f"{len(pools['pix_diag']) + len(pools['pix_off'])}, shifts {n_sh}")
        return

    # ── Metric 1: cross-classifier transfer (Δ% and absolute margin) ──────────
    if pools["lat_off"] and pools["pix_off"]:
        lat_off = np.asarray(pools["lat_off"], dtype=float)
        pix_off = np.asarray(pools["pix_off"], dtype=float)
        lat_diag = np.asarray(pools["lat_diag"], dtype=float)
        pix_diag = np.asarray(pools["pix_diag"], dtype=float)
        lat_off_a = np.asarray(pools["lat_off_abs"], dtype=float)
        pix_off_a = np.asarray(pools["pix_off_abs"], dtype=float)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        bins_pct = np.histogram_bin_edges(
            np.concatenate([lat_diag, lat_off, pix_diag, pix_off]), bins=40
        )
        axes[0].hist(lat_diag, bins=bins_pct, alpha=0.55, label="diagonal (own walk)")
        axes[0].hist(lat_off, bins=bins_pct, alpha=0.55, label="off-diagonal (transfer)")
        axes[0].set_title("Latent walks — Δ%")
        axes[1].hist(pix_diag, bins=bins_pct, alpha=0.55, label="diagonal (own walk)")
        axes[1].hist(pix_off, bins=bins_pct, alpha=0.55, label="off-diagonal (transfer)")
        axes[1].set_title("Pixel walks (control) — Δ%")
        axes[2].hist(lat_off, bins=bins_pct, alpha=0.55, label="latent off-diag")
        axes[2].hist(pix_off, bins=bins_pct, alpha=0.55, label="pixel off-diag")
        axes[2].set_title("Off-diagonal transfer — latent vs pixels")
        for ax in axes:
            ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
            ax.set_xlabel("100 · (p_t − p_s) / mean(p_s, p_t)")
            ax.set_ylabel("count")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Boundary transfer across {len(classifiers)} classifiers "
                     f"× {n_images} images (ε = {EPSILON:g})")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = os.path.join(OUT_DIR, "boundary_prob_balance.png")
        plt.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved plot: {path}")
        maybe_show(fig)

        print(f"[balance] latent  diagonal Δ%: {lat_diag.mean():+.3f} ± {lat_diag.std():.3f} "
              f"(mechanics) | diag |Δ%| ≤ 5: {(np.abs(lat_diag) <= 5).mean() * 100:.0f}% "
              f"| off-diagonal: {lat_off.mean():+.3f} ± {lat_off.std():.3f} "
              f"(min {lat_off.min():+.2f}, max {lat_off.max():+.2f}) | "
              f"off-diag |Δ%| ≤ 5: {(np.abs(lat_off) <= 5).mean() * 100:.0f}% "
              f"(ε = {EPSILON:g})")
        print(f"[balance] pixels  diagonal Δ%: {pix_diag.mean():+.3f} ± {pix_diag.std():.3f} "
              f"| diag |Δ%| ≤ 5: {(np.abs(pix_diag) <= 5).mean() * 100:.0f}% "
              f"| off-diagonal: {pix_off.mean():+.3f} ± {pix_off.std():.3f} "
              f"(min {pix_off.min():+.2f}, max {pix_off.max():+.2f}) | "
              f"off-diag |Δ%| ≤ 5: {(np.abs(pix_off) <= 5).mean() * 100:.0f}% "
              f"(ε = {EPSILON:g})")
        print(f"[margin] off-diagonal p_t − p_s: latent {lat_off_a.mean():+.4f} ± "
              f"{lat_off_a.std():.4f} | pixels {pix_off_a.mean():+.4f} ± {pix_off_a.std():.4f}")

        # — margin-shift transfer (softer criterion: shared direction?) —
        lat_sh = np.asarray(pools["lat_shift"], dtype=float)
        pix_sh = np.asarray(pools["pix_shift"], dtype=float)
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
        bins_sh = np.histogram_bin_edges(np.concatenate([lat_sh, pix_sh]), bins=40)
        axes[0].hist(lat_sh, bins=bins_sh, alpha=0.8)
        axes[0].set_title("Latent Δmargin (walk, vs recon)")
        axes[1].hist(pix_sh, bins=bins_sh, alpha=0.8)
        axes[1].set_title("Pixel Δmargin (walk, vs original)")
        for ax in (axes[0], axes[1]):
            ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
            ax.set_xlabel("Δmargin = margin(boundary) − margin(walk start)")
            ax.set_ylabel("count")
            ax.grid(alpha=0.3)
        if first_shift is not None:
            S_lat0, S_pix0, src0, tgt0, name0 = first_shift
            vmax = max(np.abs(S_lat0).max(), np.abs(S_pix0).max(), 1e-6)
            im0 = axes[2].imshow(S_lat0, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[2].set_title(f"Latent Δmargin matrix\n{name0} "
                              f"({class_names[src0]}→{class_names[tgt0]})\n"
                              f"rows=boundary of i, cols=evaluator j", fontsize=8)
            fig.colorbar(im0, ax=axes[2], fraction=0.046, pad=0.04)
            im1 = axes[3].imshow(S_pix0, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[3].set_title(f"Pixel Δmargin matrix\n{name0}\n"
                              f"rows=boundary of i, cols=evaluator j", fontsize=8)
            fig.colorbar(im1, ax=axes[3], fraction=0.046, pad=0.04)
        fig.suptitle(f"Margin-shift transfer across {len(classifiers)} classifiers "
                     f"× {n_images} images (positive = moved toward target; "
                     f"ε = {EPSILON:g})")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        shift_path = os.path.join(OUT_DIR, "boundary_transfer_shift.png")
        plt.savefig(shift_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot: {shift_path}")
        maybe_show(fig)
        print(f"[shift] latent off-diag Δmargin (walk, vs recon): "
              f"{lat_sh.mean():+.4f} ± {lat_sh.std():.4f} "
              f"| toward target: {(lat_sh > 0).mean() * 100:.0f}%")
        print(f"[shift] pixels off-diag Δmargin (walk, vs original): "
              f"{pix_sh.mean():+.4f} ± {pix_sh.std():.4f} "
              f"| toward target: {(pix_sh > 0).mean() * 100:.0f}%")
        lat_off_sh = np.asarray(pools["lat_offset"], dtype=float)
        print(f"[shift] recon vs original (latent baseline offset): "
              f"{lat_off_sh.mean():+.4f} ± {lat_off_sh.std():.4f} "
              f"| toward target: {(lat_off_sh > 0).mean() * 100:.0f}%")
        lat_S_fin = np.asarray(pools["lat_S"], dtype=float)
        pix_S_fin = np.asarray(pools["pix_S"], dtype=float)
        print(f"[pair-mass] final p_s+p_t per walk: latent {lat_S_fin.mean():.3f} ± "
              f"{lat_S_fin.std():.3f} (min {lat_S_fin.min():.3f}) | pixels "
              f"{pix_S_fin.mean():.3f} ± {pix_S_fin.std():.3f} "
              f"(min {pix_S_fin.min():.3f})")
    else:
        print("[balance] no transfer entries collected — skipping plots")

    # ── Metric 2 summary: normalised dispersion, latent vs pixel control ──────
    if var_rows:
        names  = [r[0] for r in var_rows]
        lat_r  = np.array([r[2] for r in var_rows])
        pix_r  = np.array([r[3] for r in var_rows])
        class_of = {r[0]: r[1] for r in var_rows}
        cmap = plt.get_cmap("tab10")
        colors = [cmap(class_names.index(class_of[n])) for n in names]

        x_pos = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(names)), 4.5))
        ax.bar(x_pos - 0.2, lat_r, width=0.4, color=colors, label="latent (encoder norm)")
        ax.bar(x_pos + 0.2, pix_r, width=0.4, color="lightgray",
               edgecolor="gray", label="pixels (control)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("boundary dispersion / data-spread (lower = more agreement)")
        ax.set_title(f"Normalised boundary dispersion — {len(classifiers)} classifiers")
        ax.grid(alpha=0.3, axis="y")
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(class_names))]
        handles.append(plt.Rectangle((0, 0), 1, 1, color="lightgray", ec="gray"))
        ax.legend(handles, class_names + ["pixels (control)"], fontsize=8)
        fig.tight_layout()
        var_path = os.path.join(OUT_DIR, "boundary_latent_variance.png")
        plt.savefig(var_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot: {var_path}")
        maybe_show(fig)
        print("[variance] per-image dispersion (raw | per-dim | ratio vs data spread):")
        for name, r in zip(names, var_rows):
            print(f"  {name:14s} latent {r[4]:.1f} | {r[5]:.4f} | {r[2]:.4f}   "
                  f"pixel ratio {r[3]:.4f}")
        print(f"  ratio across images — latent: {lat_r.mean():.4f} ± {lat_r.std():.4f} | "
              f"pixel: {pix_r.mean():.4f} ± {pix_r.std():.4f}")
    else:
        print("[variance] no dispersion values collected — skipping plot")

    print("\nEnsemble boundary demo complete. Plots in:", OUT_DIR)


if __name__ == "__main__":
    main()
