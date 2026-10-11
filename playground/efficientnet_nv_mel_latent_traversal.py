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

# Also add the repo root so `traversal` / core modules resolve when this
# script is run from playground/ (the GUI imports this module from there).
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from datasets import ISIC2018Dataset
from models import NVMELClassifier, NVMELAutoencoder, NVMELVAE
from utils import load_best_model, print_checkpoint_info, get_orthogonal_pca_bases
from losses import MS_SSIMLoss
from traversal import take_latent_step, take_pca_step, extrapolate_image, plot_traversal

# %%
# ── Setup & Parameters ────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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


# Traversal functions (take_latent_step, take_pca_step, extrapolate_image,
# plot_traversal) now live in the root-level traversal.py module and are
# imported above. The GUI continues to import them from this module.

# %%
if __name__ == "__main__":
    results = {}   # (label_name, k) -> traversal tuple

    for lbl_idx, label_name in enumerate(LABEL_NAMES):
        for k, (img, lbl) in enumerate(chosen_imgs[lbl_idx]):
            results[(label_name, k)] = extrapolate_image(
                img,
                lbl,
                classifier_model,
                ae_model,
                _recon_loop_loss,
                class_names=LABEL_NAMES,
                device=DEVICE,
            )

    for (label_name, k), (imgs, probs, errs, gnorms, pred_idxs, target_idx) in results.items():
        plot_traversal(
            imgs,
            probs,
            errs,
            gnorms,
            pred_idxs=pred_idxs,
            class_names=LABEL_NAMES,
            title=f"Traversal {label_name} #{k+1}",
        )

# %%
