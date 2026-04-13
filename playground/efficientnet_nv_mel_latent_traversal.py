# %%
# ── Imports ──────────────────────────────────────────────────────────────────
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from datasets import ISIC2018Dataset
from models import NVMELAutoencoder, NVMELClassifier
from utils import load_best_model, print_checkpoint_info
from losses import MS_SSIMLoss

# %%
# ── Setup & Parameters ────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

VAL_ORIG_DATASET_DIR = "dataset/ISIC_2018/ISIC2018_Task3_Validation_Input"
VAL_LABELS_CSV       = "dataset/ISIC_2018/ISIC2018_Task3_Validation_GroundTruth.csv"

AE_CHECKPOINT_DIR  = "checkpoints/efficientnet_nv_mel_ae_ms_ssim"
CLS_CHECKPOINT_DIR = "checkpoints/efficientnet_nv_mel_classifier/run_2"

IMAGE_SIZE  = 224
LABEL_NAMES = ["NV", "MEL"]

# %%
# ── Transforms ───────────────────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# %%
# ── Dataset & DataLoader ─────────────────────────────────────────────────────
val_orig_dataset = ISIC2018Dataset(
    root_dir       = VAL_ORIG_DATASET_DIR,
    transform      = val_transform,
    labels_csv     = VAL_LABELS_CSV,
    include_labels = LABEL_NAMES,
)

val_loader = DataLoader(val_orig_dataset, batch_size=1, shuffle=False)
print(f"Validation (orig) set size: {len(val_orig_dataset)}")

# %%
# ── Load Best Checkpoints ─────────────────────────────────────────────────────
ae_model = NVMELAutoencoder(freeze_up_to=0).to(DEVICE)
ae_ckpt  = load_best_model(
    ae_model,
    AE_CHECKPOINT_DIR,
    index_selector=lambda ckpt: int(np.argmin(ckpt["history"]["val_losses"])),
    device=DEVICE,
)
print("AE checkpoint info:")
print_checkpoint_info(ae_ckpt)

classifier_model = NVMELClassifier(freeze_up_to=0).to(DEVICE)
cls_ckpt = load_best_model(
    classifier_model.backbone,
    CLS_CHECKPOINT_DIR,
    index_selector=lambda ckpt: int(np.argmax(ckpt["val_aucs"])),
    device=DEVICE,
)
print("\nClassifier checkpoint info:")
print_checkpoint_info(cls_ckpt)

# %%
# ── Prepare Models for Traversal ──────────────────────────────────────────────
ae_model.eval()
classifier_model.eval()

for param in ae_model.decoder.parameters():
    param.requires_grad = True
for param in classifier_model.parameters():
    param.requires_grad = True

# %%
# ── Image Selection ───────────────────────────────────────────────────────────
N_PER_LABEL = 5   # how many images to traverse per class

chosen_imgs = {0: [], 1: []}   # label_idx -> list of (img_tensor, lbl_tensor)

for batch in val_loader:
    img = batch["image"].to(DEVICE)
    lbl = batch["label"].to(DEVICE)
    lbl_idx = lbl.argmax(dim=1).item()

    if len(chosen_imgs[lbl_idx]) < N_PER_LABEL:
        chosen_imgs[lbl_idx].append((img, lbl))

    if all(len(v) == N_PER_LABEL for v in chosen_imgs.values()):
        break


# %%
# ── Reconstruction-loop loss (MS-SSIM) ───────────────────────────────────────
_recon_loop_loss = MS_SSIMLoss(
    channels=3,
    denorm_mean=torch.tensor([0.485, 0.456, 0.406]),
    denorm_std=torch.tensor([0.229, 0.224, 0.225]),
).to(DEVICE)


def _recon_error(recon: torch.Tensor) -> float:
    """MS-SSIM error of the reconstruction loop: recon → encoder → decoder → recon'."""
    with torch.no_grad():
        z_prime    = ae_model.encoder(recon)
        recon_prime = ae_model.decoder(z_prime)
        return _recon_loop_loss(recon, recon_prime).item()


# %%
# ── Latent Traversal (unified) ────────────────────────────────────────────────
def _classifier_grad(z: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """
    Returns (G, prob, pred_idx) where G = d(logit)/dz evaluated at z.
    A fresh graph is created each call so this is safe to call in a loop.
    """
    z_ = z.detach().requires_grad_(True)
    logit = classifier_model(ae_model.decoder(z_)).squeeze(-1)
    logit.backward()
    return z_.grad.clone(), torch.sigmoid(logit).item(), int(logit.item() > 0)


def _recon_grad(z: torch.Tensor) -> torch.Tensor:
    """
    Returns V = d(err)/dz where err = MS_SSIMLoss(decoder(z), decoder(encoder(decoder(z)))).
    The encoder pass is detached so gradients only flow through the first decoder.
    """
    z_ = z.detach().requires_grad_(True)
    recon       = ae_model.decoder(z_)
    recon_prime = ae_model.decoder(ae_model.encoder(recon.detach()))
    _recon_loop_loss(recon, recon_prime.detach()).backward()
    return z_.grad.clone()


def _step_direction(z: torch.Tensor, orthogonal: bool) -> tuple[torch.Tensor, float]:
    """
    Returns (step, grad_norm) where step is the direction to move in latent space
    and grad_norm is the L2 norm of the raw classifier gradient G (before projection).
      - orthogonal=False: step = G
      - orthogonal=True : step = component of G orthogonal to V = d(err)/dz
    """
    G, _, _ = _classifier_grad(z)
    grad_norm = G.norm().item()
    if not orthogonal:
        return G, grad_norm

    V = _recon_grad(z)
    G_flat, V_flat = G.view(-1), V.view(-1)
    V_norm_sq = (V_flat * V_flat).sum()
    if V_norm_sq.item() > 1e-12:
        G_flat = G_flat - (G_flat @ V_flat) / V_norm_sq * V_flat
    return G_flat.view_as(G), grad_norm


def extrapolate_image(
    img_tensor,
    actual_label_onehot,
    steps: int = 30,
    alpha: float = 1.0,
    min_step_size: float = 1e-3,
    orthogonal: bool = False,
    target_prob: float = 0.75,
):
    """
    Traverse the latent space by following the (optionally orthogonally projected)
    classifier gradient wrt the AE latent z.

    Critically, the gradient is **recomputed at every step** so the direction
    tracks the local manifold geometry — important for a vanilla AE whose latent
    space has no smoothness incentive between in-distribution points.

    orthogonal=True: step direction is the component of G = d(logit)/dz that is
    perpendicular to V = d(err)/dz, where err = MS_SSIMLoss(recon, recon') and
    recon' = decoder(encoder(recon)).  This preserves reconstruction fidelity.

    Returns (traversal_images, traversal_probs, traversal_recon_errors).
    """
    tag        = "[Orthogonal]" if orthogonal else "[Raw]"
    actual_idx = actual_label_onehot.argmax(dim=1).item()
    img_tensor = img_tensor.to(DEVICE)
    print(f"\n{tag} Extrapolating image of actual class {LABEL_NAMES[actual_idx]}...")

    # ── Encode ────────────────────────────────────────────────────────────────
    with torch.no_grad():
        z_initial = ae_model.encoder(img_tensor)

    # ── Initial prediction info ───────────────────────────────────────────────
    G0, prob_initial, pred_initial_idx = _classifier_grad(z_initial)
    grad_norm_initial = G0.norm().item()

    with torch.no_grad():
        orig_pred_idx = int(classifier_model(img_tensor).squeeze(-1).item() > 0)

    print(f"{tag} Original Image Pred   : {LABEL_NAMES[orig_pred_idx]} (Correct: {orig_pred_idx == actual_idx})")
    print(f"{tag} Reconstructed Img Pred: {LABEL_NAMES[pred_initial_idx]} (Correct: {pred_initial_idx == actual_idx})")
    print(f"{tag} Initial AE -> Classifier prob: {prob_initial:.4f}")

    direction = 1.0 if pred_initial_idx == 0 else -1.0
    print(f"{tag} Stepping {'ALONG' if direction > 0 else 'AGAINST'} gradient "
          f"({'NV→MEL' if direction > 0 else 'MEL→NV'}).")

    # ── Traverse ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        x_recon_0 = ae_model.decoder(z_initial)

    traversal_images       = [x_recon_0.cpu().squeeze(0)]
    traversal_probs        = [prob_initial]
    traversal_recon_errors = [_recon_error(x_recon_0)]
    traversal_grad_norms   = [grad_norm_initial]

    current_z = z_initial.clone()

    for i in range(1, steps + 1):
        # Recompute direction at current latent (adapts to local manifold).
        # Step size is alpha / |G|: inversely proportional to gradient magnitude
        # so we take equal-length steps in latent space regardless of |G|.
        step, grad_norm_step = _step_direction(current_z, orthogonal)
        unit_step = step / (step.norm() + 1e-12)
        current_z = current_z + direction * max(min_step_size, alpha / (grad_norm_step + 1e-12)) * unit_step

        with torch.no_grad():
            x_recon_step = ae_model.decoder(current_z)
            prob_step    = torch.sigmoid(classifier_model(x_recon_step).squeeze(-1)).item()
            err_step     = _recon_error(x_recon_step)

        traversal_images.append(x_recon_step.cpu().squeeze(0))
        traversal_probs.append(prob_step)
        traversal_recon_errors.append(err_step)
        traversal_grad_norms.append(grad_norm_step)

        # Stop when prob has moved far enough past the decision boundary:
        # if we started as NV (prob < 0.5) we stop at prob > target_prob,
        # if we started as MEL (prob > 0.5) we stop at prob < 1 - target_prob.
        flip_threshold = target_prob if pred_initial_idx == 0 else 1.0 - target_prob
        flipped = (pred_initial_idx == 0 and prob_step >= flip_threshold) or \
                  (pred_initial_idx == 1 and prob_step <= flip_threshold)
        if flipped:
            print(f"{tag} Target prob reached at step {i}! prob={prob_step:.4f}  recon_err={err_step:.4f}  |G|={grad_norm_step:.4f}")
            break

    return traversal_images, traversal_probs, traversal_recon_errors, traversal_grad_norms


# %%
# ── Run Traversals ────────────────────────────────────────────────────────────
results = {}   # (label_name, k) -> (imgs, probs, errs, gnorms)

for lbl_idx, label_name in enumerate(LABEL_NAMES):
    for k, (img, lbl) in enumerate(chosen_imgs[lbl_idx]):
        results[(label_name, k)] = extrapolate_image(img, lbl, orthogonal=False)


# %%
# ── Visualization ─────────────────────────────────────────────────────────────
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def plot_traversal(images, probs, recon_errors, grad_norms, title="Latent Traversal"):
    n_imgs = len(images)

    # ── unnormalise all traversal frames ──────────────────────────────────────
    def _unnorm(t):
        return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    frames_rgb = [_unnorm(img) for img in images]

    # ── luminance-weighted difference map ─────────────────────────────────────
    lum_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    orig_rgb    = frames_rgb[0]
    final_rgb   = frames_rgb[-1]
    diff_map    = ((final_rgb - orig_rgb) * lum_weights).sum(axis=-1)   # H×W, range [-1, 1]
    abs_max     = max(np.abs(diff_map).max(), 1e-6)                      # keep colourmap centred

    # ── channel-preserving gamma-scaled difference overlay ────────────────────
    # Per-channel signed diff, then gamma-compress magnitude (sign handled separately
    # so negative diffs stay negative after exponentiation).
    _gamma      = 0.5
    diff_ch     = final_rgb.astype(np.float32) - orig_rgb.astype(np.float32)  # H×W×3, [-1,1]
    sign_ch     = np.sign(diff_ch)
    gamma_diff  = sign_ch * (np.abs(diff_ch) ** _gamma)                       # sign-safe gamma

    # Normalise each channel independently to [0, 1] for display.
    ch_min = gamma_diff.min(axis=(0, 1), keepdims=True)
    ch_max = gamma_diff.max(axis=(0, 1), keepdims=True)
    ch_range = np.where((ch_max - ch_min) > 1e-8, ch_max - ch_min, 1.0)
    heatmap_ch = (gamma_diff - ch_min) / ch_range                             # H×W×3, [0,1]

    # Blend with the original image: alpha=0 → orig, alpha=1 → heatmap.
    _alpha  = 0.6
    overlay = np.clip((1.0 - _alpha) * orig_rgb + _alpha * heatmap_ch, 0, 1)

    # ── layout: traversal strip (top) + comparison row (bottom) ──────────────
    # Bottom row has 4 panels; top strip spans max(n_imgs, 4) columns.
    n_top   = max(n_imgs, 4)
    fig_w   = max(3 * n_top, 12)
    fig = plt.figure(figsize=(fig_w, 7))

    # Outer grid: 2 rows.  Row 0 = traversal strip, Row 1 = compact comparison.
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.4)

    # Top row — one sub-gridspec with n_top columns
    top_gs = outer_gs[0].subgridspec(1, n_top, wspace=0.12)
    for i, (img_rgb, prob, err, gnorm) in enumerate(zip(frames_rgb, probs, recon_errors, grad_norms)):
        ax = fig.add_subplot(top_gs[0, i])
        ax.imshow(img_rgb)
        pred_label = "MEL" if prob > 0.5 else "NV"
        step_title = "Original\n(recon)" if i == 0 else f"Step {i}"
        ax.set_title(f"{step_title}\n{pred_label} ({prob:.2f})\nerr={err:.3f}  |G|={gnorm:.2f}", fontsize=8)
        ax.axis("off")

    # Bottom row — 4 compact panels in the first 4 columns.
    bot_gs = outer_gs[1].subgridspec(1, n_top, wspace=0.08)

    ax_orig = fig.add_subplot(bot_gs[0, 0])
    ax_orig.imshow(orig_rgb)
    ax_orig.set_title("Original image\n(step 0)", fontsize=9)
    ax_orig.axis("off")

    ax_final = fig.add_subplot(bot_gs[0, 1])
    ax_final.imshow(final_rgb)
    ax_final.set_title(f"Final reconstruction\n(step {n_imgs - 1})", fontsize=9)
    ax_final.axis("off")

    ax_diff = fig.add_subplot(bot_gs[0, 2])
    im = ax_diff.imshow(diff_map, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
    ax_diff.set_title("Difference map\n(lum-weighted, final − orig)", fontsize=9)
    ax_diff.axis("off")
    fig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)

    ax_overlay = fig.add_subplot(bot_gs[0, 3])
    ax_overlay.imshow(overlay)
    ax_overlay.set_title(f"Diff overlay\n(γ={_gamma}, α={_alpha})", fontsize=9)
    ax_overlay.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


for (label_name, k), (imgs, probs, errs, gnorms) in results.items():
    plot_traversal(imgs, probs, errs, gnorms, title=f"Traversal {label_name} #{k+1}")

# %%
