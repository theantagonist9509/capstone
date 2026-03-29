# %%
# ── Imports ──────────────────────────────────────────────────────────────────
import os
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from datasets import ISIC2018Dataset, TransformDataset

# %%
# ── Setup & Parameters ────────────────────────────────────────────────────────
# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

DATASET_DIR = "../dataset/ISIC_2018/ISIC2018_Task3_Training_Input"
LABELS_CSV = "../dataset/ISIC_2018/ISIC2018_Task3_Training_GroundTruth.csv"
IMAGE_SIZE = 224
LABEL_NAMES = ["NV", "MEL"]

# Transforms (using AE val_transform)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# %%
# ── Dataset & DataLoader ─────────────────────────────────────────────────────
# Load dataset
full_dataset = ISIC2018Dataset(
    root_dir=DATASET_DIR,
    transform=None,
    labels_csv=LABELS_CSV,
    include_labels=LABEL_NAMES,
)

# 20% validation split (seed 42, same as training scripts)
n_total = len(full_dataset)
n_val = int(n_total * 0.2)
n_train = n_total - n_val
generator = torch.Generator().manual_seed(42)
train_sub, val_sub = torch.utils.data.random_split(
    full_dataset, [n_train, n_val], generator=generator
)

val_dataset = TransformDataset(val_sub, val_transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print(f"Validation set size: {len(val_dataset)}")


# %%
# ── Define AE Architecture ────────────────────────────────────────────────────
# Define AE Model Architecture
class EfficientNetAutoencoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

# %%
# ── Load Best Checkpoints ─────────────────────────────────────────────────────
# Load best AE based on last checkpoint's val_losses history
ae_ckpt_dir = "../checkpoints/efficientnet_nv_mel_ae"
ae_ckpts = sorted(glob.glob(os.path.join(ae_ckpt_dir, "epoch_*.pth")))
if not ae_ckpts:
    raise FileNotFoundError("No AE checkpoints found.")
last_ae_ckpt_path = ae_ckpts[-1]
last_ae_ckpt = torch.load(last_ae_ckpt_path, map_location=DEVICE, weights_only=False)
val_losses = last_ae_ckpt["val_losses"]
best_ae_epoch = np.argmin(val_losses) + 1
best_ae_path = os.path.join(ae_ckpt_dir, f"epoch_{best_ae_epoch:03d}.pth")
print(f"Loading AE from: {best_ae_path} (val_loss: {val_losses[best_ae_epoch-1]:.4f})")
best_ae_ckpt = torch.load(best_ae_path, map_location=DEVICE, weights_only=False)

ae_backbone = models.efficientnet_b0()
ae_model = EfficientNetAutoencoder(ae_backbone).to(DEVICE)
ae_model.load_state_dict(best_ae_ckpt["model_state"])


# Load best Classifier based on last checkpoint's val_aucs history
classifier_ckpt_dir = "../checkpoints/efficientnet_nv_mel_classifier"
classifier_ckpts = sorted(glob.glob(os.path.join(classifier_ckpt_dir, "epoch_*.pth")))
if not classifier_ckpts:
    raise FileNotFoundError("No classifier checkpoints found.")
last_cls_ckpt_path = classifier_ckpts[-1]
last_cls_ckpt = torch.load(last_cls_ckpt_path, map_location=DEVICE, weights_only=False)
val_aucs = last_cls_ckpt["val_aucs"]
best_cls_epoch = np.argmax(val_aucs) + 1
best_cls_path = os.path.join(classifier_ckpt_dir, f"epoch_{best_cls_epoch:03d}.pth")
print(f"Loading Classifier from: {best_cls_path} (val_auc: {val_aucs[best_cls_epoch-1]:.4f})")
best_cls_ckpt = torch.load(best_cls_path, map_location=DEVICE, weights_only=False)

cls_backbone = models.efficientnet_b0()
in_features = cls_backbone.classifier[1].in_features
cls_backbone.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, 1),
)
classifier_model = cls_backbone.to(DEVICE)
classifier_model.load_state_dict(best_cls_ckpt["model_state"])

# ── Plot AE Training Curves ───────────────────────────────────────────────────
ae_train_losses = last_ae_ckpt["train_losses"]
ae_val_losses   = last_ae_ckpt["val_losses"]
ae_epochs_x     = range(1, len(ae_train_losses) + 1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ae_epochs_x, ae_train_losses, marker="o", linewidth=1.5, label="Train MSE")
ax.plot(ae_epochs_x, ae_val_losses,   marker="s", linewidth=1.5, label="Val MSE", linestyle="--")
ax.axvline(x=best_ae_epoch, color="red", linestyle=":", linewidth=1.5,
           label=f"Best epoch ({best_ae_epoch})")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Autoencoder – Train / Val MSE Loss")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ── Plot Classifier Training Curves ──────────────────────────────────────────
cls_train_losses = last_cls_ckpt["train_losses"]
cls_val_losses   = last_cls_ckpt["val_losses"]
cls_val_aucs     = last_cls_ckpt["val_aucs"]
cls_epochs_x     = range(1, len(cls_train_losses) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(cls_epochs_x, cls_train_losses, marker="o", linewidth=1.5, label="Train Loss")
ax1.plot(cls_epochs_x, cls_val_losses,   marker="s", linewidth=1.5, label="Val Loss", linestyle="--")
ax1.axvline(x=best_cls_epoch, color="red", linestyle=":", linewidth=1.5,
            label=f"Best epoch ({best_cls_epoch})")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title("Classifier – Train / Val Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(cls_epochs_x, cls_val_aucs, marker="s", linewidth=1.5, color="tab:green", label="Val AUC")
ax2.axvline(x=best_cls_epoch, color="red", linestyle=":", linewidth=1.5,
            label=f"Best epoch ({best_cls_epoch})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("ROC-AUC")
ax2.set_ylim(0, 1)
ax2.set_title("Classifier – Validation AUC")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.6)

plt.suptitle("Classifier Training Curves", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# ── Prepare Models for Traversal ──────────────────────────────────────────────
# Prepare models for traversal
ae_model.eval()
classifier_model.eval()

# "set requires_grad to true for 1. the classifier 2. the decoder (including the latent representation)"
for param in ae_model.decoder.parameters():
    param.requires_grad = True
for param in classifier_model.parameters():
    param.requires_grad = True

criterion = nn.BCEWithLogitsLoss()

# %%
# ── Evaluate Classifier on AE Reconstructions (Val Set) ───────────────────────
import sklearn.metrics as metrics

orig_labels = []
orig_preds  = []
orig_probs  = []
recon_preds = []
recon_probs = []

print("Evaluating classifier on originals & AE reconstructions...")
with torch.no_grad():
    for imgs, lbls in tqdm(val_loader, desc="Eval"):
        imgs = imgs.to(DEVICE)
        true_labels = lbls.argmax(dim=1)

        # ── Original images ───────────────────────────────────────────────────
        orig_logits = classifier_model(imgs).squeeze(-1)
        orig_prob   = torch.sigmoid(orig_logits)
        orig_pred   = (orig_prob > 0.5).int()

        # ── AE Reconstructions ────────────────────────────────────────────────
        recons      = ae_model(imgs)
        rec_logits  = classifier_model(recons).squeeze(-1)
        rec_prob    = torch.sigmoid(rec_logits)
        rec_pred    = (rec_prob > 0.5).int()

        orig_labels.extend(true_labels.cpu().numpy())
        orig_preds.extend(orig_pred.cpu().numpy())
        orig_probs.extend(orig_prob.cpu().numpy())
        recon_preds.extend(rec_pred.cpu().numpy())
        recon_probs.extend(rec_prob.cpu().numpy())

orig_labels = np.array(orig_labels)
orig_preds  = np.array(orig_preds)
orig_probs  = np.array(orig_probs)
recon_preds = np.array(recon_preds)
recon_probs = np.array(recon_probs)

def _eval_metrics(y_true, y_pred, y_prob):
    cm        = metrics.confusion_matrix(y_true, y_pred)
    acc       = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall    = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1        = metrics.f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = metrics.roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")
    return cm, acc, precision, recall, f1, roc_auc

orig_cm,  orig_acc,  orig_prec,  orig_rec,  orig_f1,  orig_auc  = _eval_metrics(orig_labels, orig_preds,  orig_probs)
recon_cm, recon_acc, recon_prec, recon_rec, recon_f1, recon_auc = _eval_metrics(orig_labels, recon_preds, recon_probs)

print("\n─────────────────────────────────────────────────────────────")
print(f"{'Metric':<12}{'Original':>12}{'Reconstruction':>16}")
print("─────────────────────────────────────────────────────────────")
print(f"{'Accuracy':<12}{orig_acc:>12.4f}{recon_acc:>16.4f}")
print(f"{'Precision':<12}{orig_prec:>12.4f}{recon_prec:>16.4f}")
print(f"{'Recall':<12}{orig_rec:>12.4f}{recon_rec:>16.4f}")
print(f"{'F1':<12}{orig_f1:>12.4f}{recon_f1:>16.4f}")
print(f"{'ROC AUC':<12}{orig_auc:>12.4f}{recon_auc:>16.4f}")
print("─────────────────────────────────────────────────────────────\n")
print(f"Confusion Matrix (Original):\n{orig_cm}")
print(f"\nConfusion Matrix (Reconstruction):\n{recon_cm}\n")

# ── Confusion Matrix Heatmaps ─────────────────────────────────────────────────
import matplotlib.colors as mcolors

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

for ax, cm, title in [
    (ax1, orig_cm,  "Original Images"),
    (ax2, recon_cm, "AE Reconstructions"),
]:
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion Matrix\n({title})", fontsize=11)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1]); ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(LABEL_NAMES)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight="bold")

plt.tight_layout()
plt.show()

# %%
# ── Image Selection ───────────────────────────────────────────────────────────
# Select an image to traverse
# Let's pick an NV and a MEL
chosen_img_nv = None
chosen_lbl_nv = None
chosen_img_mel = None
chosen_lbl_mel = None

for img, lbl in val_loader:
    img = img.to(DEVICE)
    lbl = lbl.to(DEVICE)
    lbl_idx = lbl.argmax(dim=1).item()
    
    if lbl_idx == 0 and chosen_img_nv is None:
        chosen_img_nv = img
        chosen_lbl_nv = lbl
    elif lbl_idx == 1 and chosen_img_mel is None:
        chosen_img_mel = img
        chosen_lbl_mel = lbl
        
    if chosen_img_nv is not None and chosen_img_mel is not None:
        break


# %%
# ── Latent Extrapolation Method ───────────────────────────────────────────────
def extrapolate_image(img_tensor, actual_label_onehot, steps=30, alpha=2.0):
    """
    Traverse the latent space starting from img.
    """
    actual_idx = actual_label_onehot.argmax(dim=1).item()
    img_tensor = img_tensor.to(DEVICE)
    
    print(f"\nExtrapolating image of actual class {LABEL_NAMES[actual_idx]}...")
    
    # 1. Encode image
    with torch.no_grad():
        z_initial = ae_model.encoder(img_tensor)
        
    # We compute gradients w.r.t the latent vector
    z = z_initial.clone().detach().requires_grad_(True)
    
    # Forward pass: Decode then classify
    x_recon = ae_model.decoder(z)
    logits = classifier_model(x_recon).squeeze(-1)
    
    # Compute initial prediction
    prob_initial = torch.sigmoid(logits).item()
    pred_initial_idx = int(logits.item() > 0)
    
    # Inform on correctness
    with torch.no_grad():
        orig_logits = classifier_model(img_tensor).squeeze(-1)
        orig_pred_idx = int(orig_logits.item() > 0)
        
    print(f"Original Image Pred: {LABEL_NAMES[orig_pred_idx]} (Correct: {orig_pred_idx == actual_idx})")
    print(f"Reconstructed Image Pred: {LABEL_NAMES[pred_initial_idx]} (Correct: {pred_initial_idx == actual_idx})")
    print(f"Initial AE -> Classifier prob: {prob_initial:.4f}")
    
    # Backpropagate the logit to get gradient directly pointing towards MEL (class 1)
    logits.backward()
    gradient = z.grad.clone()
    
    # Determine step direction
    # If pred is 0 (NV), we want to push towards 1 (MEL) -> step ALONG gradient to increase logit
    # If pred is 1 (MEL), we want to push towards 0 (NV) -> step AGAINST gradient to decrease logit
    if pred_initial_idx == 0:
        direction = 1.0  
        print("Current pred is NV (0). Stepping ALONG gradient to increase prediction to MEL (1).")
    else:
        direction = -1.0
        print("Current pred is MEL (1). Stepping AGAINST gradient to decrease prediction to NV (0).")

    traversal_images = [x_recon.detach().cpu().squeeze(0)]
    traversal_probs = [prob_initial]
    
    current_z = z_initial.clone()
    
    for i in range(1, steps + 1):
        # Step in the latent space
        current_z = current_z + direction * alpha * gradient
        
        # Decode and classify
        with torch.no_grad():
            x_recon_step = ae_model.decoder(current_z)
            logits_step = classifier_model(x_recon_step).squeeze(-1)
            prob_step = torch.sigmoid(logits_step).item()
            
        traversal_images.append(x_recon_step.cpu().squeeze(0))
        traversal_probs.append(prob_step)
        
        # Check if prediction flipped
        if (prob_step > 0.5) != (prob_initial > 0.5):
            print(f"Prediction flipped at step {i}! Probability: {prob_step:.4f}")
            break
            
    return traversal_images, traversal_probs

# %%
# ── Run Extrapolation ─────────────────────────────────────────────────────────
if chosen_img_nv is not None:
    imgs_nv, probs_nv = extrapolate_image(chosen_img_nv, chosen_lbl_nv)
    
if chosen_img_mel is not None:
    imgs_mel, probs_mel = extrapolate_image(chosen_img_mel, chosen_lbl_mel, alpha=5)

# %%
# ── Visualization ─────────────────────────────────────────────────────────────
# Unnormalize and plot
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def plot_traversal(images, probs, title="Latent Traversal"):
    n_imgs = len(images)
    fig, axes = plt.subplots(1, n_imgs, figsize=(3 * n_imgs, 3))
    if n_imgs == 1:
        axes = [axes]
    
    for i, (ax, img, prob) in enumerate(zip(axes, images, probs)):
        img_unnorm = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img_unnorm)
        pred_label = "MEL" if prob > 0.5 else "NV"
        step_title = "Original" if i == 0 else f"Step {i}"
        ax.set_title(f"{step_title}\n{pred_label} ({prob:.2f})")
        ax.axis("off")
        
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(title.replace(" ", "_").lower() + ".png")
    print(f"Saved plot successfully to {title.replace(' ', '_').lower()}.png")

if chosen_img_nv is not None:
    plot_traversal(imgs_nv, probs_nv, title="Traversal from NV Image")
    
if chosen_img_mel is not None:
    plot_traversal(imgs_mel, probs_mel, title="Traversal from MEL Image")

# %%
