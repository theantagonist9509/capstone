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
# **Random baselines.** For every gradient walk (latent and pixel) the script
# also draws N random offsets (default 10) from the SAME start point, at the
# same straight-line distance as that walk (‖z_final − z_start‖ in latent
# space, ‖x_final − x_start‖ in normalised pixel space), in a uniform random
# direction. Pixel offsets are clamped to the valid image range, exactly like
# the pixel walk (so a clamped offset can be slightly shorter). Every random
# point goes through the same stats as the gradient boundary images, averaged
# over the N draws. These show how much of each result comes from the
# gradient direction rather than from the distance moved.
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
# - locally: `python runs/shapes/traverse.py [--smoke] [--arch conv_vae] [--orthogonal]
#   [--n-random 10] [--random-seed 42]`
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
from matplotlib.patches import Patch
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

# Valid pixel range in normalised space (same as extrapolate_pixels' clamp).
PIX_LO = (0.0 - MEAN) / STD
PIX_HI = (1.0 - MEAN) / STD

# Plot colours per method.
C_LAT_GRAD, C_LAT_RAND = "tab:blue", "tab:cyan"
C_PIX_GRAD, C_PIX_RAND = "tab:orange", "tab:green"

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


def _batch_prob(logits, class_idx):
    """
    P(class_idx) for a BATCH of logits, mirroring traversal._prob_of:
    K logits → the softmax column; 1 logit → sigmoid for class 1 and
    1 − sigmoid for class 0.
    """
    if logits.shape[-1] == 1:
        p1 = torch.sigmoid(logits.view(-1))
        return p1 if class_idx == 1 else 1.0 - p1
    return torch.softmax(logits, dim=-1)[:, class_idx]


def cross_eval_batch(frames, classifiers, src, tgt):
    """Like cross_eval, but each entry of `frames` is a batch (n,3,H,W) of
    random draws for classifier i's walk. Each C×C entry is the mean over the
    n draws. Returns (M, A, S): symmetric % difference, absolute margin
    p_target − p_source, and pair mass p_source + p_target.
    """
    C = len(classifiers)
    M = np.zeros((C, C))
    A = np.zeros((C, C))
    S = np.zeros((C, C))
    with torch.no_grad():
        for i in range(C):
            for j, clf_j in enumerate(classifiers):
                logits = clf_j(frames[i])
                p_s = _batch_prob(logits, src)
                p_t = _batch_prob(logits, tgt)
                A[i, j] = float((p_t - p_s).mean())
                M[i, j] = float((100.0 * (p_t - p_s)
                                 / (0.5 * (p_s + p_t)).clamp_min(1e-8)).mean())
                S[i, j] = float((p_s + p_t).mean())
    return M, A, S


def random_offsets(origin, dist, n, gen, lo=None, hi=None):
    """
    n points at straight-line distance `dist` from `origin` (shape (1, ...)),
    each in a uniform random direction (normalised Gaussian vector). If lo/hi
    are given the points are clamped to [lo, hi] (so they may land a little
    closer than `dist`).
    """
    u = torch.randn((n,) + tuple(origin.shape[1:]), generator=gen, device=origin.device)
    norm = u.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *([1] * (u.dim() - 1)))
    pts = origin + dist * u / norm
    if lo is not None:
        pts = pts.clamp(lo, hi)
    return pts


def _overlay(ax, series, bins, title, xlabel):
    """Overlay histograms. series: [(data, label, colour or None)]."""
    for data, label, color in series:
        ax.hist(data, bins=bins, alpha=0.55, label=label, color=color)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def _table_row(name, diag, off, off_abs, shift, S, disp):
    return (f"{name:<16s} {diag.mean():+8.2f} ±{diag.std():6.2f} | "
            f"{off.mean():+8.2f} ±{off.std():6.2f} | "
            f"{(np.abs(off) <= 5).mean() * 100:4.0f}% | {off_abs.mean():+8.4f} | "
            f"{shift.mean():+8.4f} ({(shift > 0).mean() * 100:3.0f}%) | "
            f"{S.mean():.3f} | {disp:.4f}")


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
    parser.add_argument("--orthogonal", action="store_true",
                        help="project the latent walk off the reconstruction-error "
                             "gradient (orthogonal / paper traversal); "
                             "default is the vanilla score-gradient walk")
    parser.add_argument("--classifier-seeds", type=int, nargs="+", default=None,
                        metavar="SEED",
                        help="restrict the ensemble to these seed_<seed> dirs "
                             "(default: every seed dir found)")
    parser.add_argument("--n-random", type=int, default=10,
                        help="random draws per walk for the random baselines "
                             "(averaged; default: 10)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="seed for the random baselines (default: 42)")
    args = parser.parse_args()

    n_rand = max(1, args.n_random)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(args.random_seed)

    class_names, val_sub = build_val_split()
    print(f"Val split size: {len(val_sub)} | classes: {class_names}")
    print(f"Traversing arch: {args.arch}"
          + (" (VAE latent = mu, posterior mean)" if args.arch == "conv_vae" else "")
          + f" | mode: {'orthogonal' if args.orthogonal else 'vanilla'}"
          + f" | ε = {EPSILON:g} | random baselines: {n_rand} draws, seed {args.random_seed}")

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

    # "r" prefix = random baseline. lat_/pix_ = gradient walks.
    pools = {k: [] for k in (
        "lat_diag", "lat_off", "pix_diag", "pix_off",
        "lat_off_abs", "pix_off_abs",
        "lat_shift", "pix_shift", "lat_offset",
        "lat_S", "pix_S",
        "rlat_diag", "rlat_off", "rpix_diag", "rpix_off",
        "rlat_off_abs", "rpix_off_abs",
        "rlat_shift", "rpix_shift",
        "rlat_S", "rpix_S",
        "lat_dist", "pix_dist", "rpix_keep",
    )}
    first_shift = None   # dict of the first image's shift matrices
    var_rows = []        # one dict per image (dispersion results)
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
                    steps=STEPS, orthogonal=args.orthogonal,
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
                z0 = ae_model.encoder(x)             # latent walk start
                x_recon = ae_model.decoder(z0)
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

            # — random baselines: same start, same straight-line distance —
            # One batch of n_rand random points per classifier walk i, at the
            # distance walk i moved. Latent points are decoded; pixel points
            # are clamped to the valid range like the pixel walk.
            rnd_lat_x, rnd_lat_z, rnd_pix_x = [], [], []
            with torch.no_grad():
                for i in range(C):
                    d_lat = float((lat_zs[i] - z0).norm())
                    d_pix = float((pix_xs[i] - x).norm())
                    pools["lat_dist"].append(d_lat)
                    pools["pix_dist"].append(d_pix)

                    zr = random_offsets(z0, d_lat, n_rand, gen)
                    rnd_lat_z.append(zr.flatten(1))
                    rnd_lat_x.append(ae_model.decoder(zr))

                    xr = random_offsets(x, d_pix, n_rand, gen, PIX_LO, PIX_HI)
                    rnd_pix_x.append(xr)
                    if d_pix > 1e-12:
                        kept = float((xr - x).flatten(1).norm(dim=1).mean()) / d_pix
                        pools["rpix_keep"].append(kept)

            M_rl, A_rl, S_rl = cross_eval_batch(rnd_lat_x, classifiers, src, tgt)
            M_rp, A_rp, S_rp = cross_eval_batch(rnd_pix_x, classifiers, src, tgt)
            pools["rlat_diag"].extend(np.diag(M_rl))
            pools["rpix_diag"].extend(np.diag(M_rp))
            pools["rlat_off"].extend(M_rl[off])
            pools["rpix_off"].extend(M_rp[off])
            pools["rlat_off_abs"].extend(A_rl[off])
            pools["rpix_off_abs"].extend(A_rp[off])
            S_rlat = A_rl - margins_recon[None, :]
            S_rpix = A_rp - margins_orig[None, :]
            pools["rlat_shift"].extend(S_rlat[off])
            pools["rpix_shift"].extend(S_rpix[off])
            pools["rlat_S"].extend(np.diag(S_rl))
            pools["rpix_S"].extend(np.diag(S_rp))

            if first_shift is None:
                first_shift = {
                    "lat": S_lat.copy(), "pix": S_pix.copy(),
                    "rlat": S_rlat.copy(), "rpix": S_rpix.copy(),
                    "src": src, "tgt": tgt, "name": f"{label_name}#{k+1}",
                }

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

            # Random baselines: spread over classifiers within each draw,
            # averaged over the draws.
            Zr = torch.stack(rnd_lat_z).float()                      # (C, n, D)
            rlat_var = float(((Zr - Zr.mean(dim=0, keepdim=True)) ** 2).sum(dim=-1).mean())
            Xr = torch.stack([t.flatten(1) for t in rnd_pix_x]).float()  # (C, n, 3HW)
            rpix_var = float(((Xr - Xr.mean(dim=0, keepdim=True)) ** 2).sum(dim=-1).mean())
            rlat_ratio = rlat_var / max(latent_ref_var, 1e-12)
            rpix_ratio = rpix_var / max(pixel_ref_var, 1e-12)

            var_rows.append({
                "name": f"{label_name}#{k+1}", "cls": label_name,
                "lat": lat_ratio, "pix": pix_ratio,
                "rlat": rlat_ratio, "rpix": rpix_ratio,
                "lat_var": lat_var, "lat_pd": lat_var_per_dim, "pix_var": pix_var,
            })
            print(f"  dispersion: latent var {lat_var:.1f} "
                  f"(per-dim {lat_var_per_dim:.4f}, ratio {lat_ratio:.4f}) | "
                  f"pixel var {pix_var:.1f} (ratio {pix_ratio:.4f})")

    if args.smoke:
        n_lat = len(pools["lat_diag"]) + len(pools["lat_off"])
        n_sh  = len(pools["lat_shift"]) + len(pools["pix_shift"])
        n_r   = (len(pools["rlat_diag"]) + len(pools["rlat_off"])
                 + len(pools["rpix_diag"]) + len(pools["rpix_off"]))
        print(f"\n[SMOKE] {n_images} image × {len(classifiers)} classifiers ok | "
              f"transfer entries: latent {n_lat}, pixels "
              f"{len(pools['pix_diag']) + len(pools['pix_off'])}, shifts {n_sh}, "
              f"random baselines {n_r}")
        return

    # ── Random-baseline distances ─────────────────────────────────────────────
    lat_dist = np.asarray(pools["lat_dist"], dtype=float)
    pix_dist = np.asarray(pools["pix_dist"], dtype=float)
    keep = np.asarray(pools["rpix_keep"], dtype=float)
    print(f"\n[random] matched straight-line distance per walk: latent "
          f"{lat_dist.mean():.3f} ± {lat_dist.std():.3f} | pixels "
          f"{pix_dist.mean():.3f} ± {pix_dist.std():.3f} "
          f"(clamping kept {keep.mean() * 100:.1f}% of the pixel offset)"
          if keep.size else
          f"\n[random] matched distance: latent {lat_dist.mean():.3f} | "
          f"pixels {pix_dist.mean():.3f}")

    # ── Metric 1: cross-classifier transfer (Δ% and absolute margin) ──────────
    if pools["lat_off"] and pools["pix_off"]:
        P = {k: np.asarray(v, dtype=float) for k, v in pools.items()}
        lat_off, pix_off = P["lat_off"], P["pix_off"]
        lat_diag, pix_diag = P["lat_diag"], P["pix_diag"]
        lat_off_a, pix_off_a = P["lat_off_abs"], P["pix_off_abs"]
        rlat_off, rpix_off = P["rlat_off"], P["rpix_off"]
        rlat_diag, rpix_diag = P["rlat_diag"], P["rpix_diag"]

        # Shared bins so every panel is comparable.
        bins_pct = np.histogram_bin_edges(
            np.concatenate([lat_diag, lat_off, pix_diag, pix_off,
                            rlat_diag, rlat_off, rpix_diag, rpix_off]), bins=40
        )
        xl_pct = "100 · (p_t − p_s) / mean(p_s, p_t)"

        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        # Row 1: gradient walks
        _overlay(axes[0][0], [(lat_diag, "diagonal (own walk)", None),
                              (lat_off, "off-diagonal (transfer)", None)],
                 bins_pct, "Latent gradient walks — Δ%", xl_pct)
        _overlay(axes[0][1], [(pix_diag, "diagonal (own walk)", None),
                              (pix_off, "off-diagonal (transfer)", None)],
                 bins_pct, "Pixel gradient walks (control) — Δ%", xl_pct)
        _overlay(axes[0][2], [(lat_off, "latent off-diag", C_LAT_GRAD),
                              (pix_off, "pixel off-diag", C_PIX_GRAD)],
                 bins_pct, "Gradient off-diagonal — latent vs pixels", xl_pct)
        # Row 2: random baselines
        _overlay(axes[1][0], [(rlat_diag, "diagonal (own walk's offset)", None),
                              (rlat_off, "off-diagonal (transfer)", None)],
                 bins_pct, f"Latent random offsets ({n_rand} draws) — Δ%", xl_pct)
        _overlay(axes[1][1], [(rpix_diag, "diagonal (own walk's offset)", None),
                              (rpix_off, "off-diagonal (transfer)", None)],
                 bins_pct, f"Pixel random offsets ({n_rand} draws) — Δ%", xl_pct)
        _overlay(axes[1][2], [(rlat_off, "latent random off-diag", C_LAT_RAND),
                              (rpix_off, "pixel random off-diag", C_PIX_RAND)],
                 bins_pct, "Random off-diagonal — latent vs pixels", xl_pct)
        # Row 3: gradient vs random
        _overlay(axes[2][0], [(lat_off, "gradient", C_LAT_GRAD),
                              (rlat_off, "random", C_LAT_RAND)],
                 bins_pct, "Latent off-diagonal — gradient vs random", xl_pct)
        _overlay(axes[2][1], [(pix_off, "gradient", C_PIX_GRAD),
                              (rpix_off, "random", C_PIX_RAND)],
                 bins_pct, "Pixel off-diagonal — gradient vs random", xl_pct)
        _overlay(axes[2][2], [(lat_off, "latent gradient", C_LAT_GRAD),
                              (rlat_off, "latent random", C_LAT_RAND),
                              (pix_off, "pixel gradient", C_PIX_GRAD),
                              (rpix_off, "pixel random", C_PIX_RAND)],
                 bins_pct, "Off-diagonal — all four", xl_pct)
        fig.suptitle(f"Boundary transfer across {len(classifiers)} classifiers "
                     f"× {n_images} images (ε = {EPSILON:g})")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
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
        print(f"[balance] latent  RANDOM diagonal Δ%: {rlat_diag.mean():+.3f} ± "
              f"{rlat_diag.std():.3f} | diag |Δ%| ≤ 5: "
              f"{(np.abs(rlat_diag) <= 5).mean() * 100:.0f}% | off-diagonal: "
              f"{rlat_off.mean():+.3f} ± {rlat_off.std():.3f} | off-diag |Δ%| ≤ 5: "
              f"{(np.abs(rlat_off) <= 5).mean() * 100:.0f}%")
        print(f"[balance] pixels  RANDOM diagonal Δ%: {rpix_diag.mean():+.3f} ± "
              f"{rpix_diag.std():.3f} | diag |Δ%| ≤ 5: "
              f"{(np.abs(rpix_diag) <= 5).mean() * 100:.0f}% | off-diagonal: "
              f"{rpix_off.mean():+.3f} ± {rpix_off.std():.3f} | off-diag |Δ%| ≤ 5: "
              f"{(np.abs(rpix_off) <= 5).mean() * 100:.0f}%")
        print(f"[margin] off-diagonal p_t − p_s: latent {lat_off_a.mean():+.4f} ± "
              f"{lat_off_a.std():.4f} | pixels {pix_off_a.mean():+.4f} ± {pix_off_a.std():.4f}"
              f" | latent random {P['rlat_off_abs'].mean():+.4f} ± "
              f"{P['rlat_off_abs'].std():.4f} | pixels random "
              f"{P['rpix_off_abs'].mean():+.4f} ± {P['rpix_off_abs'].std():.4f}")

        # — margin-shift transfer (softer criterion: shared direction?) —
        lat_sh, pix_sh = P["lat_shift"], P["pix_shift"]
        rlat_sh, rpix_sh = P["rlat_shift"], P["rpix_shift"]
        bins_sh = np.histogram_bin_edges(
            np.concatenate([lat_sh, pix_sh, rlat_sh, rpix_sh]), bins=40
        )
        xl_sh = "Δmargin = margin(point) − margin(walk start)"
        fig, axes = plt.subplots(2, 4, figsize=(18, 8.6))
        _overlay(axes[0][0], [(lat_sh, "gradient", C_LAT_GRAD),
                              (rlat_sh, "random", C_LAT_RAND)],
                 bins_sh, "Latent Δmargin (vs recon) — gradient vs random", xl_sh)
        _overlay(axes[0][1], [(pix_sh, "gradient", C_PIX_GRAD),
                              (rpix_sh, "random", C_PIX_RAND)],
                 bins_sh, "Pixel Δmargin (vs original) — gradient vs random", xl_sh)
        _overlay(axes[1][0], [(lat_sh, "latent gradient", C_LAT_GRAD),
                              (pix_sh, "pixel gradient", C_PIX_GRAD)],
                 bins_sh, "Gradient Δmargin — latent vs pixels", xl_sh)
        _overlay(axes[1][1], [(rlat_sh, "latent random", C_LAT_RAND),
                              (rpix_sh, "pixel random", C_PIX_RAND)],
                 bins_sh, "Random Δmargin — latent vs pixels", xl_sh)
        if first_shift is not None:
            fs = first_shift
            vmax = max(np.abs(fs[m]).max() for m in ("lat", "pix", "rlat", "rpix"))
            vmax = max(vmax, 1e-6)
            pair = f"{class_names[fs['src']]}→{class_names[fs['tgt']]}"
            panels = [(axes[0][2], "lat", "Latent gradient"),
                      (axes[0][3], "pix", "Pixel gradient"),
                      (axes[1][2], "rlat", "Latent random"),
                      (axes[1][3], "rpix", "Pixel random")]
            for ax, key, name in panels:
                im = ax.imshow(fs[key], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax.set_title(f"{name} Δmargin matrix\n{fs['name']} ({pair})\n"
                             f"rows=point of i, cols=evaluator j", fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            for ax in (axes[0][2], axes[0][3], axes[1][2], axes[1][3]):
                ax.axis("off")
        fig.suptitle(f"Margin-shift transfer across {len(classifiers)} classifiers "
                     f"× {n_images} images (positive = moved toward target; "
                     f"ε = {EPSILON:g})")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
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
        print(f"[shift] latent RANDOM off-diag Δmargin: "
              f"{rlat_sh.mean():+.4f} ± {rlat_sh.std():.4f} "
              f"| toward target: {(rlat_sh > 0).mean() * 100:.0f}%")
        print(f"[shift] pixels RANDOM off-diag Δmargin: "
              f"{rpix_sh.mean():+.4f} ± {rpix_sh.std():.4f} "
              f"| toward target: {(rpix_sh > 0).mean() * 100:.0f}%")
        lat_off_sh = P["lat_offset"]
        print(f"[shift] recon vs original (latent baseline offset): "
              f"{lat_off_sh.mean():+.4f} ± {lat_off_sh.std():.4f} "
              f"| toward target: {(lat_off_sh > 0).mean() * 100:.0f}%")
        lat_S_fin, pix_S_fin = P["lat_S"], P["pix_S"]
        print(f"[pair-mass] final p_s+p_t per walk: latent {lat_S_fin.mean():.3f} ± "
              f"{lat_S_fin.std():.3f} (min {lat_S_fin.min():.3f}) | pixels "
              f"{pix_S_fin.mean():.3f} ± {pix_S_fin.std():.3f} "
              f"(min {pix_S_fin.min():.3f}) | latent random "
              f"{P['rlat_S'].mean():.3f} ± {P['rlat_S'].std():.3f} "
              f"(min {P['rlat_S'].min():.3f}) | pixels random "
              f"{P['rpix_S'].mean():.3f} ± {P['rpix_S'].std():.3f} "
              f"(min {P['rpix_S'].min():.3f})")
    else:
        P = None
        print("[balance] no transfer entries collected — skipping plots")

    # ── Metric 2 summary: normalised dispersion, latent vs pixel control ──────
    if var_rows:
        names  = [r["name"] for r in var_rows]
        lat_r  = np.array([r["lat"] for r in var_rows])
        pix_r  = np.array([r["pix"] for r in var_rows])
        rlat_r = np.array([r["rlat"] for r in var_rows])
        rpix_r = np.array([r["rpix"] for r in var_rows])
        class_of = {r["name"]: r["cls"] for r in var_rows}
        cmap = plt.get_cmap("tab10")
        colors = [cmap(class_names.index(class_of[n])) for n in names]

        x_pos = np.arange(len(names))
        w = 0.2
        fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(names)), 4.8))
        ax.bar(x_pos - 1.5 * w, lat_r, width=w, color=colors)
        ax.bar(x_pos - 0.5 * w, rlat_r, width=w, color=colors,
               hatch="//", edgecolor="white", alpha=0.6)
        ax.bar(x_pos + 0.5 * w, pix_r, width=w, color="lightgray", edgecolor="gray")
        ax.bar(x_pos + 1.5 * w, rpix_r, width=w, color="lightgray",
               edgecolor="gray", hatch="//", alpha=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("boundary dispersion / data-spread (lower = more agreement)")
        ax.set_title(f"Normalised boundary dispersion — {len(classifiers)} classifiers "
                     f"(random = mean of {n_rand} draws)")
        ax.grid(alpha=0.3, axis="y")
        handles = [Patch(color=cmap(i)) for i in range(len(class_names))]
        handles += [Patch(facecolor="lightgray", edgecolor="gray"),
                    Patch(facecolor="white", edgecolor="gray", hatch="//")]
        ax.legend(handles,
                  class_names + ["pixels (grey, solid = gradient)",
                                 "random offsets (hatched)"], fontsize=8)
        fig.tight_layout()
        var_path = os.path.join(OUT_DIR, "boundary_latent_variance.png")
        plt.savefig(var_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot: {var_path}")
        maybe_show(fig)
        print("[variance] per-image dispersion (raw | per-dim | ratio vs data spread):")
        for r in var_rows:
            print(f"  {r['name']:14s} latent {r['lat_var']:.1f} | {r['lat_pd']:.4f} | "
                  f"{r['lat']:.4f}   pixel ratio {r['pix']:.4f}   "
                  f"random: latent {r['rlat']:.4f}, pixel {r['rpix']:.4f}")
        print(f"  ratio across images — latent: {lat_r.mean():.4f} ± {lat_r.std():.4f} | "
              f"pixel: {pix_r.mean():.4f} ± {pix_r.std():.4f} | "
              f"latent random: {rlat_r.mean():.4f} ± {rlat_r.std():.4f} | "
              f"pixel random: {rpix_r.mean():.4f} ± {rpix_r.std():.4f}")
    else:
        print("[variance] no dispersion values collected — skipping plot")
        lat_r = pix_r = rlat_r = rpix_r = None

    # ── Summary table: all four methods ───────────────────────────────────────
    if P is not None and lat_r is not None:
        print("\n[summary] all four methods (pooled over images × classifiers)")
        print(f"{'method':<16s} {'diag Δ% (mean ± sd)':<18s} | "
              f"{'off Δ% (mean ± sd)':<18s} | {'|Δ%|≤5':>5s} | "
              f"{'off margin':>10s} | {'shift (toward)':<17s} | "
              f"{'S':<5s} | disp")
        print(_table_row("latent gradient", P["lat_diag"], P["lat_off"],
                         P["lat_off_abs"], P["lat_shift"], P["lat_S"], lat_r.mean()))
        print(_table_row("latent random", P["rlat_diag"], P["rlat_off"],
                         P["rlat_off_abs"], P["rlat_shift"], P["rlat_S"], rlat_r.mean()))
        print(_table_row("pixel gradient", P["pix_diag"], P["pix_off"],
                         P["pix_off_abs"], P["pix_shift"], P["pix_S"], pix_r.mean()))
        print(_table_row("pixel random", P["rpix_diag"], P["rpix_off"],
                         P["rpix_off_abs"], P["rpix_shift"], P["rpix_S"], rpix_r.mean()))

    print("\nEnsemble boundary demo complete. Plots in:", OUT_DIR)


if __name__ == "__main__":
    main()
