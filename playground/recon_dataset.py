# %%
# ── Imports ──────────────────────────────────────────────────────────────────
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from datasets import ISIC2018Dataset
from models import NVMELAutoencoder
from utils import load_best_model, print_checkpoint_info

print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

# %%
# ── Parameters (edit here) ────────────────────────────────────────────────────
DATASET_DIR    = "./dataset/ISIC_2018/ISIC2018_Task3_Training_Input"
LABELS_CSV     = "./dataset/ISIC_2018/ISIC2018_Task3_Training_GroundTruth.csv"
IMAGE_SIZE     = 224          # EfficientNet-B0 default
BATCH_SIZE     = 16
NUM_WORKERS    = 4
CHECKPOINT_DIR = "./checkpoints/efficientnet_nv_mel_ae_ms_ssim"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_NAMES         = ["NV", "MEL"]

RECON_DIR      = "./dataset/ISIC_2018/ISIC2018_Task3_Training_Input_Recon_MS_SSIM"

os.makedirs(RECON_DIR, exist_ok=True)

print(f"Using device  : {DEVICE}")
print(f"Image size    : {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Checkpoint dir: {os.path.abspath(CHECKPOINT_DIR)}")

# %%
# ── Transforms ───────────────────────────────────────────────────────────────
imagenet_mean   = torch.tensor([0.485, 0.456, 0.406])
imagenet_std    = torch.tensor([0.229, 0.224, 0.225])

input_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

output_transform = transforms.Compose([
    transforms.Normalize(-imagenet_mean / imagenet_std, 1 / imagenet_std),
])

print("Transform pipelines defined.")

# %%
# ── Dataset & DataLoaders ─────────────────────────────────────────────────────
dataset = ISIC2018Dataset(
    root_dir       = DATASET_DIR,
    transform      = input_transform,
    labels_csv     = LABELS_CSV,
    include_labels = LABEL_NAMES,
)

print(f"Total labeled samples (NV+MEL): {len(dataset)}")

loader = DataLoader(
    dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = DEVICE == "cuda",
)

print(f"batches/epoch : {len(loader)}")

# %%
# ── Sanity check: visualise a batch ──────────────────────────────────────────
batch = next(iter(loader))
sample_imgs     = batch["image"]
sample_labels   = batch["label"]
print("Batch shape :", sample_imgs.shape)
print("Labels (raw):", sample_labels[:8].tolist())

fig, axes = plt.subplots(1, 6, figsize=(15, 3))
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
for i, ax in enumerate(axes):
    img = sample_imgs[i].permute(1, 2, 0).numpy()
    img = (img * std + mean).clip(0, 1)
    cls = sample_labels[i].argmax().item()
    ax.imshow(img)
    ax.set_title(LABEL_NAMES[cls], fontsize=9)
    ax.axis("off")
plt.suptitle("Sample images from train DataLoader", fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Load best model
model = NVMELAutoencoder(freeze_up_to=0).to(DEVICE)
ckpt = load_best_model(model, CHECKPOINT_DIR, lambda ckpt: np.argmin(np.array(ckpt["history"]["val_losses"])), DEVICE)
print_checkpoint_info(ckpt)

# %%
# Generate reconstructions and write to disk
model.eval()
with torch.no_grad():
    for batch in tqdm(loader, desc=f"Reconstructing images", leave=False):
        imgs = batch["image"].to(DEVICE, non_blocking=True)
        
        recons = model(imgs)
        output_recons = output_transform(recons)

        for i, recon in enumerate(output_recons):
            save_image(recon, f"{RECON_DIR}/{batch['id'][i]}.png")
# %%
