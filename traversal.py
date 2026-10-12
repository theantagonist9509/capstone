# %%
# ── Latent Space Traversal (generalized: binary 1-logit + multiclass K-logit) ─
#
# Scoring convention, decided at runtime from the classifier's output shape:
#   - 1 logit  (legacy binary): sigmoid; G = ∇_z logit; direction ±1 decides
#     which class to move toward (identical math to the original playground
#     traversal implementation).
#   - K logits (softmax):       pred = argmax; score toward class t from
#     current class c is logit_t − logit_c. With target=None ("auto"), t is
#     the second-best class (the most competitive competitor).
#
# All functions take models explicitly; nothing here is tied to a dataset.

import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from utils import get_orthogonal_pca_bases, maybe_show

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# %%
def predict(classifier_model: nn.Module, x: torch.Tensor):
    """
    Single-image prediction.

    Returns (logits, probs, pred_idx):
      - 1-logit classifier: logits (1,1), probs shape (1,) = [sigmoid(logit)],
        pred_idx = int(logit > 0).
      - K-logit classifier: logits (1,K), probs shape (K,) = softmax,
        pred_idx = argmax.
    """
    logits = classifier_model(x)
    if logits.shape[-1] == 1:
        logit = logits.view(-1)[0]
        probs = torch.sigmoid(logit).view(1)
        pred_idx = int((logit > 0).item())
    else:
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(probs.argmax().item())
    return logits, probs, pred_idx


def resolve_target(logits: torch.Tensor, pred_idx: int, target=None) -> int:
    """
    Resolves the target class index.
      - target is an int → used as-is.
      - target is None / "auto" → second-best class (for 1-logit binary:
        the other class, i.e. 1 - pred_idx).
    """
    if target is not None and target != "auto":
        return int(target)
    if logits.shape[-1] == 1:
        return 1 - pred_idx
    return int(logits.view(-1).argsort(descending=True)[1].item())


def score(logits: torch.Tensor, pred_idx: int, target_idx: int) -> torch.Tensor:
    """
    Scalar score whose increase moves the prediction toward target_idx.
      - 1 logit: the logit itself (direction ±1 is applied by the caller;
        for binary, stepping along the logit moves toward class 1).
      - K logits: logit_target − logit_current (one-vs-rest margin).
    """
    if logits.shape[-1] == 1:
        return logits.view(-1)[0]
    return logits.view(-1)[target_idx] - logits.view(-1)[pred_idx]


# %%
def classifier_grad(
    z: torch.Tensor,
    classifier_model: nn.Module,
    ae_model: nn.Module,
    target=None,
):
    """
    G = ∇_z score(decoder(z)) evaluated at z, with a fresh graph.

    Returns (G, probs, pred_idx, target_idx). For 1-logit classifiers G is
    always ∇_z logit (the legacy behavior); the caller applies direction ±1.
    For K-logit classifiers G = ∇_z (logit_t − logit_c) already points toward
    the target.
    """
    z_ = z.detach().requires_grad_(True)
    logits = classifier_model(ae_model.decoder(z_))

    if logits.shape[-1] == 1:
        logit = logits.view(-1)[0]
        probs = torch.sigmoid(logit).view(1)
        pred_idx = int((logit > 0).item())
        target_idx = resolve_target(logits, pred_idx, target)
        logit.backward()
    else:
        probs = torch.softmax(logits, dim=-1)[0]
        pred_idx = int(probs.argmax().item())
        target_idx = resolve_target(logits, pred_idx, target)
        (logits.view(-1)[target_idx] - logits.view(-1)[pred_idx]).backward()

    return z_.grad.clone(), probs, pred_idx, target_idx


# %%
def recon_error(ae_model: nn.Module, recon: torch.Tensor, recon_loss_fn: nn.Module) -> float:
    """
    Error of the reconstruction loop: recon → encoder → decoder → recon'.
    """
    with torch.no_grad():
        z_prime     = ae_model.encoder(recon)
        recon_prime = ae_model.decoder(z_prime)
        return recon_loss_fn(recon, recon_prime).item()


# %%
def take_latent_step(
    z: torch.Tensor,
    classifier_model: nn.Module,
    ae_model: nn.Module,
    recon_loss_fn: nn.Module,
    direction: float,
    optimizer: torch.optim.Optimizer,
    orthogonal: bool = False,
    target_class: int = None,
):
    """
    Takes a single optimization step in the latent space using the provided
    optimizer. The tensor `z` must be the parameter optimized by `optimizer`
    and have `requires_grad=True`.

    1-logit classifier: loss = -direction * logit (legacy behavior; direction
    is +1.0 to move toward class 1 / MEL, -1.0 toward class 0 / NV).

    K-logit classifier: loss = -direction * (logit_t − logit_c), where
    c = argmax and t = target_class (or the second-best class when None).

    orthogonal=True: before applying the step, the score gradient is
    orthogonally projected relative to V = d(err)/dz, where
    err = recon_loss_fn(recon, recon'), recon' = decoder(encoder(recon)).
    """
    optimizer.zero_grad()
    logits = classifier_model(ae_model.decoder(z))

    if logits.shape[-1] == 1:
        logit = logits.view(-1)[0]
        loss  = -direction * logit
    else:
        logits = logits.view(-1)
        pred_idx = int(logits.argmax().item())
        target_idx = resolve_target(logits, pred_idx, target_class)
        loss = -direction * (logits[target_idx] - logits[pred_idx])

    loss.backward()

    if orthogonal:
        z_ = z.detach().requires_grad_(True)
        recon       = ae_model.decoder(z_)
        recon_prime = ae_model.decoder(ae_model.encoder(recon.detach()))
        recon_loss_fn(recon, recon_prime.detach()).backward()
        V = z_.grad.clone()

        G_flat, V_flat = z.grad.view(-1), V.view(-1)
        V_norm_sq = (V_flat * V_flat).sum()
        if V_norm_sq.item() > 1e-12:
            G_flat = G_flat - (G_flat @ V_flat) / V_norm_sq * V_flat
        z.grad = G_flat.view_as(z.grad)

    optimizer.step()
    return z


# %%
def take_pca_step(
    z: torch.Tensor,
    classifier_model: nn.Module,
    ae_model: nn.Module,
    all_embeddings: torch.Tensor,
    base_idx: int,
    sign: float,
    lr: float,
    target_class: int = None,
):
    """
    Takes a single orthogonal PCA step. `z` is returned as a new detached
    tensor. The reference vector for the orthogonal complement is the
    classifier score gradient at the current point (logit gradient for
    1-logit classifiers, score gradient otherwise).
    """
    z_ = z.detach().requires_grad_(True)
    logits = classifier_model(ae_model.decoder(z_))
    if logits.shape[-1] == 1:
        logits.view(-1)[0].backward()
    else:
        logits = logits.view(-1)
        pred_idx = int(logits.argmax().item())
        target_idx = resolve_target(logits, pred_idx, target_class)
        (logits[target_idx] - logits[pred_idx]).backward()
    ref_grad = z_.grad.clone().detach()
    z_ = z_.detach()

    bases = get_orthogonal_pca_bases(all_embeddings, ref_grad, k=2)
    base = bases[base_idx].to(z.device)

    return z_ + sign * lr * base


# %%
def extrapolate_image(
    img_tensor,
    actual_label_onehot,
    classifier_model: nn.Module,
    ae_model: nn.Module,
    recon_loss_fn: nn.Module,
    class_names: list,
    device,
    steps: int = 30,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs=None,
    orthogonal: bool = True,
    target_prob: float = 0.75,
    target="auto",
):
    """
    Traverse the latent space by optimizing the latent vector z to move the
    classifier's prediction toward the target class.

    The gradient is recomputed at every step so the direction tracks the local
    manifold geometry. orthogonal=True projects each score gradient orthogonal
    to V = d(err)/dz (err = recon loop loss) to preserve reconstruction
    fidelity.

    target="auto" (default): the second-best class w.r.t. the initial
    prediction — for 1-logit binary classifiers this is exactly the legacy
    behavior (step along the logit for NV, against it for MEL). An explicit
    class index targets that class instead.

    Returns (traversal_images, traversal_probs, traversal_recon_errors,
    traversal_grad_norms, traversal_pred_idxs, target_idx) where
    traversal_probs holds P(target) at each step.
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {"lr": 0.05}

    tag = "[Orthogonal]" if orthogonal else "[Raw]"
    actual_idx = int(actual_label_onehot.argmax().item())  # works for (K,) or (1,K)
    img_tensor = img_tensor.to(device)
    print(f"\n{tag} Extrapolating image of actual class {class_names[actual_idx]}...")

    # ── Encode ────────────────────────────────────────────────────────────────
    with torch.no_grad():
        z_initial = ae_model.encoder(img_tensor)

    # ── Initial prediction info ───────────────────────────────────────────────
    G0, probs_initial, pred_initial_idx, target_idx = classifier_grad(
        z_initial, classifier_model, ae_model, target
    )
    grad_norm_initial = G0.norm().item()
    is_binary_logit = probs_initial.numel() == 1

    with torch.no_grad():
        _, orig_probs, orig_pred_idx = predict(classifier_model, img_tensor)

    def _fmt_probs(p):
        return "[" + ", ".join(f"{v:.2f}" for v in p.view(-1).tolist()) + "]"

    print(f"{tag} Original Image Pred   : {class_names[orig_pred_idx]} (Correct: {orig_pred_idx == actual_idx})")
    print(f"{tag} Reconstructed Img Pred: {class_names[pred_initial_idx]} (Correct: {pred_initial_idx == actual_idx})")
    print(f"{tag} Initial probs (recon) : {_fmt_probs(probs_initial)}")
    print(f"{tag} Traversing toward     : {class_names[target_idx]}")

    # Direction: for 1-logit classifiers ±1 on the logit (legacy); for K-logit
    # classifiers the score is already oriented toward the target.
    if is_binary_logit:
        direction = 1.0 if target_idx == 1 else -1.0
        print(f"{tag} Stepping {'ALONG' if direction > 0 else 'AGAINST'} the logit gradient.")
    else:
        direction = 1.0
        print(f"{tag} Stepping +1 along the score gradient (logit_t − logit_c).")

    # ── Traverse ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        x_recon_0 = ae_model.decoder(z_initial)

    traversal_images       = [x_recon_0.cpu().squeeze(0)]
    traversal_probs        = [probs_initial[target_idx].item()]
    traversal_recon_errors = [recon_error(ae_model, x_recon_0, recon_loss_fn)]
    traversal_grad_norms   = [grad_norm_initial]
    traversal_pred_idxs    = [pred_initial_idx]

    current_z = z_initial.detach().clone()
    current_z.requires_grad = True

    optimizer = optimizer_cls([current_z], **optimizer_kwargs)

    for i in range(1, steps + 1):
        take_latent_step(
            current_z,
            classifier_model,
            ae_model,
            recon_loss_fn,
            direction,
            optimizer,
            orthogonal,
            target_class=target_idx,
        )

        grad_norm_step = current_z.grad.norm().item() if current_z.grad is not None else 0.0

        with torch.no_grad():
            x_recon_step = ae_model.decoder(current_z)
            _, probs_step, pred_step_idx = predict(classifier_model, x_recon_step)
            err_step = recon_error(ae_model, x_recon_step, recon_loss_fn)

        traversal_images.append(x_recon_step.cpu().squeeze(0))
        traversal_probs.append(probs_step[target_idx].item())
        traversal_recon_errors.append(err_step)
        traversal_grad_norms.append(grad_norm_step)
        traversal_pred_idxs.append(pred_step_idx)

        # Stop when the prediction has moved far enough past the boundary:
        # 1-logit: P(target) past the legacy flip-threshold logic.
        # K-logit: predicted class is the target AND P(target) >= target_prob.
        if is_binary_logit:
            p_target = probs_step[target_idx].item()
            flip_threshold = target_prob if pred_initial_idx == 0 else 1.0 - target_prob
            flipped = (pred_initial_idx == 0 and p_target >= flip_threshold) or \
                      (pred_initial_idx == 1 and p_target <= flip_threshold)
        else:
            flipped = (pred_step_idx == target_idx
                       and probs_step[target_idx].item() >= target_prob)

        if flipped:
            print(f"{tag} Target prob reached at step {i}! "
                  f"p({class_names[target_idx]})={probs_step[target_idx].item():.4f}  "
                  f"recon_err={err_step:.4f}  |G|={grad_norm_step:.4f}")
            break

    return (traversal_images, traversal_probs, traversal_recon_errors,
            traversal_grad_norms, traversal_pred_idxs, target_idx)


# %%
def plot_traversal(
    images,
    probs,
    recon_errors,
    grad_norms,
    pred_idxs=None,
    class_names=None,
    title="Latent Traversal",
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    save_path=None,
    show=True,
):
    """
    Traversal strip + 4-panel comparison (orig recon, final recon, luminance
    difference map, gamma-scaled difference overlay).

    probs holds P(target) per step. pred_idxs holds the predicted class index
    per step; when given (with class_names), frame labels show the predicted
    class name, otherwise they fall back to the legacy binary "MEL"/"NV"
    thresholding on prob.
    """
    n_imgs = len(images)

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)

    # ── unnormalise all traversal frames ──────────────────────────────────────
    def _unnorm(t):
        return (t * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()

    frames_rgb = [_unnorm(img) for img in images]

    # ── luminance-weighted difference map ─────────────────────────────────────
    lum_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    orig_rgb    = frames_rgb[0]
    final_rgb   = frames_rgb[-1]
    diff_map    = ((final_rgb - orig_rgb) * lum_weights).sum(axis=-1)   # H×W, range [-1, 1]
    abs_max     = max(np.abs(diff_map).max(), 1e-6)                      # keep colourmap centred

    # ── channel-preserving gamma-scaled difference overlay ────────────────────
    _gamma     = 0.5
    diff_ch    = final_rgb.astype(np.float32) - orig_rgb.astype(np.float32)  # H×W×3, [-1,1]
    sign_ch    = np.sign(diff_ch)
    gamma_diff = sign_ch * (np.abs(diff_ch) ** _gamma)                       # sign-safe gamma

    ch_min = gamma_diff.min(axis=(0, 1), keepdims=True)
    ch_max = gamma_diff.max(axis=(0, 1), keepdims=True)
    ch_range = np.where((ch_max - ch_min) > 1e-8, ch_max - ch_min, 1.0)
    heatmap_ch = (gamma_diff - ch_min) / ch_range                             # H×W×3, [0,1]

    _alpha  = 0.6
    overlay = np.clip((1.0 - _alpha) * orig_rgb + _alpha * heatmap_ch, 0, 1)

    # ── layout: traversal strip (top) + comparison row (bottom) ──────────────
    n_top = max(n_imgs, 4)
    fig_w = max(3 * n_top, 12)
    fig = plt.figure(figsize=(fig_w, 7))

    outer_gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.4)

    top_gs = outer_gs[0].subgridspec(1, n_top, wspace=0.12)
    for i, (img_rgb, prob, err, gnorm) in enumerate(zip(frames_rgb, probs, recon_errors, grad_norms)):
        ax = fig.add_subplot(top_gs[0, i])
        ax.imshow(img_rgb)
        if pred_idxs is not None and class_names is not None:
            pred_label = class_names[pred_idxs[i]]
        else:
            pred_label = "MEL" if prob > 0.5 else "NV"
        step_title = "Original\n(recon)" if i == 0 else f"Step {i}"
        ax.set_title(f"{step_title}\n{pred_label} ({prob:.2f})\nerr={err:.3f}  |G|={gnorm:.2f}", fontsize=8)
        ax.axis("off")

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
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot: {save_path}")
    maybe_show(fig, show)


# %%
