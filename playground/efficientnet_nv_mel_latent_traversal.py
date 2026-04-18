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

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from datasets import ISIC2018Dataset
from models import NVMELAutoencoder, NVMELClassifier
from utils import load_best_model, print_checkpoint_info, get_orthogonal_pca_bases
from losses import MS_SSIMLoss

# %%
# ── Setup & Parameters ────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INTERACTIVE = "--interactive" in sys.argv
print(f"Using device: {DEVICE}")

VAL_ORIG_DATASET_DIR = "dataset/ISIC_2018/ISIC2018_Task3_Validation_Input"
VAL_LABELS_CSV       = "dataset/ISIC_2018/ISIC2018_Task3_Validation_GroundTruth.csv"

AE_CHECKPOINT_DIR  = "checkpoints/efficientnet_nv_mel_ae_ms_ssim"
CLS_CHECKPOINT_DIR = "checkpoints/efficientnet_nv_mel_classifier_recon_ms_ssim"

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
    load_into_memory = True,
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
    classifier_model,
    CLS_CHECKPOINT_DIR,
    index_selector=lambda ckpt: int(np.argmax(ckpt["history"]["val_recon_f1s"])),
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


def extrapolate_image(
    img_tensor,
    actual_label_onehot,
    steps: int = 30,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs=None,
    orthogonal: bool = False,
    target_prob: float = 0.75,
):
    """
    Traverse the latent space by optimizing the latent vector z using a
    configurable PyTorch optimizer to extremize the classifier logit.

    Critically, the gradient is **recomputed at every step** so the direction
    tracks the local manifold geometry — important for a vanilla AE whose latent
    space has no smoothness incentive between in-distribution points.

    orthogonal=True: Before applying the optimizer step, the gradient of the logit 
    is orthogonally projected relative to V = d(err)/dz, where err = MS_SSIMLoss(recon, recon').
    This preserves reconstruction fidelity.

    Returns (traversal_images, traversal_probs, traversal_recon_errors, traversal_grad_norms).
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {"lr": 0.05}

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

    current_z = z_initial.detach().clone()
    current_z.requires_grad = True

    optimizer = optimizer_cls([current_z], **optimizer_kwargs)

    for i in range(1, steps + 1):
        optimizer.zero_grad()

        # Compute logit for current_z
        logit = classifier_model(ae_model.decoder(current_z)).squeeze(-1)
        
        # We want logit to go UP if direction == 1, reducing the loss -logit.
        # We want logit to go DOWN if direction == -1, reducing the loss logit.
        loss = -direction * logit
        loss.backward()

        grad_norm_step = current_z.grad.norm().item()

        if orthogonal:
            V = _recon_grad(current_z)
            G_flat, V_flat = current_z.grad.view(-1), V.view(-1)
            V_norm_sq = (V_flat * V_flat).sum()
            if V_norm_sq.item() > 1e-12:
                G_flat = G_flat - (G_flat @ V_flat) / V_norm_sq * V_flat
            current_z.grad = G_flat.view_as(current_z.grad)

        optimizer.step()

        with torch.no_grad():
            x_recon_step = ae_model.decoder(current_z)
            prob_step    = torch.sigmoid(classifier_model(x_recon_step).squeeze(-1)).item()
            err_step     = _recon_error(x_recon_step)

        traversal_images.append(x_recon_step.cpu().squeeze(0))
        traversal_probs.append(prob_step)
        traversal_recon_errors.append(err_step)
        traversal_grad_norms.append(grad_norm_step)

        # Stop when prob has moved far enough past the decision boundary:
        flip_threshold = target_prob if pred_initial_idx == 0 else 1.0 - target_prob
        flipped = (pred_initial_idx == 0 and prob_step >= flip_threshold) or \
                  (pred_initial_idx == 1 and prob_step <= flip_threshold)
        if flipped:
            print(f"{tag} Target prob reached at step {i}! prob={prob_step:.4f}  recon_err={err_step:.4f}  |G|={grad_norm_step:.4f}")
            break

    return traversal_images, traversal_probs, traversal_recon_errors, traversal_grad_norms

# %%
# ── Run Traversals ────────────────────────────────────────────────────────────

if INTERACTIVE:
    import gradio as gr
    import matplotlib.cm as cm
    def launch_gradio_app():
        print("Preparing gallery images and computing dataset embeddings...")
        mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        gallery_items = []
        all_imgs = []
        with torch.no_grad():
            for idx in range(len(val_orig_dataset)):
                item = val_orig_dataset[idx]
                img_t = item["image"]
                lbl_t = item["label"]
                img_disp = (img_t * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
                lbl_idx = lbl_t.argmax().item()
                gallery_items.append((img_disp, f"{LABEL_NAMES[lbl_idx]}"))
                
                all_imgs.append(img_t)
                
            all_imgs_batch = torch.stack(all_imgs, dim=0).to(DEVICE)
            all_embeddings = ae_model.encoder(all_imgs_batch)

        with gr.Blocks() as demo:
            gr.Markdown("# Latent Traversal Interactive Interface")
            
            current_z_state = gr.State(None)
            orig_recon_state = gr.State(None)
            
            with gr.Tab("Selection & Traversal"):
                gallery = gr.Gallery(value=gallery_items, label="Validation Dataset (Click to Select & Start)", columns=8, allow_preview=False)
                
                with gr.Row():
                    btn_step_mel = gr.Button("Step Towards MEL (Gradient IN)")
                    btn_step_nv = gr.Button("Step Towards NV (Gradient AGAINST)")
                    
                with gr.Row():
                    btn_pca_pos_1 = gr.Button("Step +PCA Base 1")
                    btn_pca_neg_1 = gr.Button("Step -PCA Base 1")
                    btn_pca_pos_2 = gr.Button("Step +PCA Base 2")
                    btn_pca_neg_2 = gr.Button("Step -PCA Base 2")
                
                with gr.Row():
                    chk_orthogonal = gr.Checkbox(label="Orthogonal", value=False)
                    lr_slider = gr.Slider(0.001, 10.0, value=1.0, label="Learning Rate (Step Size)")
                    n_steps_slider = gr.Slider(1, 100, value=1, step=1, label="Number of Steps")
                    
                with gr.Row():
                    orig_img_display = gr.Image(label="Selected Original Image")
                    orig_recon_display = gr.Image(label="Original Reconstruction")
                    out_img = gr.Image(label="Current Decoded Image")
                    diff_map_display = gr.Image(label="Difference Map (recon vs current)")
                    
                out_info = gr.Textbox(label="Current Info")
                
                def get_diff_map(orig_rgb, current_rgb):
                    lum_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
                    diff_map = ((current_rgb - orig_rgb) * lum_weights).sum(axis=-1)
                    abs_max = max(np.abs(diff_map).max(), 1e-6)
                    norm_diff = (diff_map + abs_max) / (2 * abs_max)
                    return cm.RdBu_r(norm_diff)[:, :, :3]
                
                def on_gallery_select(evt: gr.SelectData):
                    idx = evt.index
                    item = val_orig_dataset[idx]
                    img_t = item["image"].to(DEVICE)
                    lbl_t = item["label"]
                    lbl_idx = lbl_t.argmax().item()
                    
                    with torch.no_grad():
                        z_initial = ae_model.encoder(img_t.unsqueeze(0))
                        x_recon = ae_model.decoder(z_initial)
                    
                    logit = classifier_model(x_recon).squeeze(-1)
                    prob = torch.sigmoid(logit).item()
                    err = _recon_error(x_recon)
                    
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
                    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
                    
                    orig_disp = (img_t * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    sq_disp = (x_recon[0] * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    
                    info_text = f"Class: {LABEL_NAMES[lbl_idx]} | Prob MEL: {prob:.4f} | Recon Err: {err:.4f}"
                    
                    diff_colored = get_diff_map(sq_disp, sq_disp)
                    
                    return orig_disp, sq_disp, sq_disp, diff_colored, info_text, z_initial.detach(), sq_disp
                    
                gallery.select(on_gallery_select, inputs=[], outputs=[orig_img_display, orig_recon_display, out_img, diff_map_display, out_info, current_z_state, orig_recon_state])
                
                def take_step(z, lr, n_steps, orthogonal, direction_mode, orig_recon_disp):
                    if z is None:
                        return None, None, "Please select an image first.", None
                    
                    z_ = z.detach().clone()
                    
                    for _ in range(n_steps):
                        z_ = z_.requires_grad_(True)
                        logit = classifier_model(ae_model.decoder(z_)).squeeze(-1)
                        loss = -direction_mode * logit
                        loss.backward()
                        
                        g = z_.grad.clone()
                        if orthogonal:
                            V = _recon_grad(z_)
                            g_flat, V_flat = g.view(-1), V.view(-1)
                            V_norm_sq = (V_flat * V_flat).sum()
                            if V_norm_sq.item() > 1e-12:
                                g_flat = g_flat - (g_flat @ V_flat) / V_norm_sq * V_flat
                            g = g_flat.view_as(g)
                            
                        z_ = (z_.detach() - lr * g).detach()
                        
                    new_z = z_
                    
                    with torch.no_grad():
                        x_recon = ae_model.decoder(new_z)
                        new_logit = classifier_model(x_recon).squeeze(-1)
                        prob = torch.sigmoid(new_logit).item()
                        err = _recon_error(x_recon)
                        
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
                    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
                    sq_disp = (x_recon[0] * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    
                    diff_colored = get_diff_map(orig_recon_disp, sq_disp)
                    
                    return sq_disp, diff_colored, f"Prob MEL: {prob:.4f} | Recon Err: {err:.4f}", new_z
                    
                def take_pca_step(z, lr, n_steps, base_idx, sign, orig_recon_disp):
                    if z is None:
                        return None, None, "Please select an image first.", None
                    
                    z_ = z.detach().clone()
                    
                    for _ in range(n_steps):
                        z_ = z_.requires_grad_(True)
                        logit = classifier_model(ae_model.decoder(z_)).squeeze(-1)
                        logit.backward()
                        ref_grad = z_.grad.clone().detach()
                        z_ = z_.detach()
                        
                        bases = get_orthogonal_pca_bases(all_embeddings, ref_grad, k=2)
                        base = bases[base_idx].to(DEVICE)
                        
                        z_ = z_ + sign * lr * base
                        
                    new_z = z_
                    
                    with torch.no_grad():
                        x_recon = ae_model.decoder(new_z)
                        new_logit = classifier_model(x_recon).squeeze(-1)
                        prob = torch.sigmoid(new_logit).item()
                        err = _recon_error(x_recon)
                        
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
                    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
                    sq_disp = (x_recon[0] * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    
                    diff_colored = get_diff_map(orig_recon_disp, sq_disp)
                    
                    return sq_disp, diff_colored, f"Prob MEL: {prob:.4f} | Recon Err: {err:.4f}", new_z
                    
                btn_step_mel.click(take_step, inputs=[current_z_state, lr_slider, n_steps_slider, chk_orthogonal, gr.State(1.0), orig_recon_state], outputs=[out_img, diff_map_display, out_info, current_z_state])
                btn_step_nv.click(take_step, inputs=[current_z_state, lr_slider, n_steps_slider, chk_orthogonal, gr.State(-1.0), orig_recon_state], outputs=[out_img, diff_map_display, out_info, current_z_state])
                
                btn_pca_pos_1.click(take_pca_step, inputs=[current_z_state, lr_slider, n_steps_slider, gr.State(0), gr.State(1.0), orig_recon_state], outputs=[out_img, diff_map_display, out_info, current_z_state])
                btn_pca_neg_1.click(take_pca_step, inputs=[current_z_state, lr_slider, n_steps_slider, gr.State(0), gr.State(-1.0), orig_recon_state], outputs=[out_img, diff_map_display, out_info, current_z_state])
                btn_pca_pos_2.click(take_pca_step, inputs=[current_z_state, lr_slider, n_steps_slider, gr.State(1), gr.State(1.0), orig_recon_state], outputs=[out_img, diff_map_display, out_info, current_z_state])
                btn_pca_neg_2.click(take_pca_step, inputs=[current_z_state, lr_slider, n_steps_slider, gr.State(1), gr.State(-1.0), orig_recon_state], outputs=[out_img, diff_map_display, out_info, current_z_state])

        demo.launch(server_name="0.0.0.0", share=False)
            
    launch_gradio_app()
    import sys
    sys.exit(0)

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

if not INTERACTIVE:
    results = {}   # (label_name, k) -> (imgs, probs, errs, gnorms)

    for lbl_idx, label_name in enumerate(LABEL_NAMES):
        for k, (img, lbl) in enumerate(chosen_imgs[lbl_idx]):
            results[(label_name, k)] = extrapolate_image(img, lbl, orthogonal=False)

    for (label_name, k), (imgs, probs, errs, gnorms) in results.items():
        plot_traversal(imgs, probs, errs, gnorms, title=f"Traversal {label_name} #{k+1}")

# %%
