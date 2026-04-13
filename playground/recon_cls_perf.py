# %% [markdown]
# # Image Reconstruction Classification Performance

# %%
# ── Imports ──────────────────────────────────────────────────────────────────
import os
import sys
import glob
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from datasets import ISIC2018Dataset, TransformDataset
from models import NVMELClassifier, NVMELAutoencoder
from utils import load_best_model, print_checkpoint_info

print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

# %%
# ── Parameters (edit here) ────────────────────────────────────────────────────
DATASET_DIR         = "dataset/ISIC_2018/ISIC2018_Task3_Training_Input"
LABELS_CSV          = "dataset/ISIC_2018/ISIC2018_Task3_Training_GroundTruth.csv"
IMAGE_SIZE          = 224          # EfficientNet-B0 default
BATCH_SIZE          = 8
NUM_WORKERS         = 2
VAL_SPLIT           = 0.2          # fraction held out for validation
CLS_CHECKPOINT_DIR  = "checkpoints/efficientnet_nv_mel_classifier/run_2"
AE_CHECKPOINT_DIR   = "checkpoints/efficientnet_nv_mel_ae_ms_ssim"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_NAMES         = ["NV", "MEL"]

print(f"Using device  : {DEVICE}")

# %%
# ── Dataset & DataLoaders ─────────────────────────────────────────────────────
# Base dataset with transform=None so it returns raw PIL images.
# TransformDataset wraps each split with its own augmentation pipeline.

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

full_dataset = ISIC2018Dataset(
    root_dir       = DATASET_DIR,
    transform      = None,
    labels_csv     = LABELS_CSV,
    include_labels = LABEL_NAMES,
)

print(f"Total labeled samples (NV+MEL): {len(full_dataset)}")

# ── Train / Val split ─────────────────────────────────────────────────────────
n_total = len(full_dataset)
n_val   = int(n_total * VAL_SPLIT)
n_train = n_total - n_val

generator = torch.Generator().manual_seed(42)
train_sub, val_sub = random_split(
    full_dataset, [n_train, n_val], generator=generator
)

val_dataset = TransformDataset(val_sub, val_transform)

print(f"Val samples : {len(val_dataset):,}")

val_loader = DataLoader(
    val_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = DEVICE == "cuda",
)

print(f"Val batches/epoch : {len(val_loader)}")

# %%
# Load best classifier
classifier = NVMELClassifier(freeze_up_to=0).to(DEVICE)
ckpt = load_best_model(classifier.backbone, CLS_CHECKPOINT_DIR, lambda ckpt: np.argmax(np.array(ckpt["val_aucs"])), DEVICE)

print("Classifier:")
print_checkpoint_info(ckpt)

# %%
# Load best AE
ae = NVMELAutoencoder(freeze_up_to=0).to(DEVICE)
ckpt = load_best_model(ae, AE_CHECKPOINT_DIR, lambda ckpt: np.argmin(np.array(ckpt["history"]["val_losses"])), DEVICE)

print("Autoencoder:")
print_checkpoint_info(ckpt)

# %%
all_logits  = []
all_targets = []   # ground-truth class indices

with torch.no_grad():
    for batch in tqdm(val_loader, desc=f"Classifier AE Reconstructions", leave=False):
        imgs          = batch["image"].to(DEVICE, non_blocking=True)
        labels_onehot = batch["label"].to(DEVICE, non_blocking=True)
        labels        = labels_onehot.argmax(dim=1)

        recons = ae(imgs)
        logits = classifier(recons).squeeze(-1)

        all_logits.extend(logits.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

all_preds = (np.array(all_logits) > 0).astype(int)
val_auc  = roc_auc_score(all_targets, all_logits)
val_acc  = accuracy_score(all_targets, all_preds)
val_prec = precision_score(all_targets, all_preds, zero_division=0)
val_rec  = recall_score(all_targets, all_preds, zero_division=0)
val_f1   = f1_score(all_targets, all_preds, zero_division=0)
val_cm   = confusion_matrix(all_targets, all_preds)

print(f"Classifier on Reconstructed Images:")
print(f"    Val AUC: {val_auc:.4f}")
print(f"    Val accuracy: {val_acc:.4f}")
print(f"    Val precision: {val_prec:.4f}")
print(f"    Val recall: {val_rec:.4f}")
print(f"    Val F1: {val_f1:.4f}")
print(f"    Val confusion matrix:")
print(val_cm)

# %%