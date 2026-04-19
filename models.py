# %%
import torch
import torch.nn as nn
from torchvision import transforms, models

# %%
# ── Model – EfficientNet-B0 Classifier ────────────────────────────
#
# EfficientNet-B0 feature extractor layout (backbone.features):
#   [0]  Conv2dNormActivation  – stem (32 filters)
#   [1]  Sequential            – MBConv block stage 1
#   [2]  Sequential            – MBConv block stage 2
#   [3]  Sequential            – MBConv block stage 3
#   [4]  Sequential            – MBConv block stage 4
#   [5]  Sequential            – MBConv block stage 5
#   [6]  Sequential            – MBConv block stage 6
#   [7]  Sequential            – MBConv block stage 7
#   [8]  Conv2dNormActivation  – head conv
#
# Backbone i/o: backbone.features -> backbone.avgpool -> backbone.classifier

class NVMELClassifier(nn.Module):
    def __init__(self, freeze_up_to):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for i in range(freeze_up_to): # inlcudes stem
            for param in self.backbone.features[i].parameters():
                param.requires_grad = False

        # Replace the classifier head: 1280-d → 1 (NV/MEL)
        in_features = self.backbone.classifier[1].in_features
        dropout_rate = 0.4
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x)

class GrayscaleNVMELClassifier(nn.Module):
    def __init__(self, freeze_up_to):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Sum first layer weights accross channels
        conv_orig = self.backbone.features[0][0]
        w_orig = conv_orig.weight

        conv_new = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        conv_new.weight.data = w_orig.sum(dim=1, keepdim=True)
        self.backbone.features[0][0] = conv_new

        for i in range(freeze_up_to): # inlcudes stem
            for param in self.backbone.features[i].parameters():
                param.requires_grad = False

        # Replace the classifier head: 1280-d → 1 (NV/MEL)
        in_features = self.backbone.classifier[1].in_features
        dropout_rate = 0.4
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x)

# %%
# ── Model – EfficientNet-B0 Autoencoder ───────────────────────────────────────
class NVMELAutoencoder(nn.Module):
    def __init__(self, freeze_up_to):
        super().__init__()

        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for i in range(freeze_up_to): # includes stem
            for param in backbone.features[i].parameters():
                param.requires_grad = False

        self.encoder = backbone.features
        # encoder output: (B, 1280, 7, 7) for 224x224 input
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

class GrayscaleNVMELAutoencoder(nn.Module):
    def __init__(self, freeze_up_to):
        super().__init__()

        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Sum first layer weights accross channels
        conv_orig = backbone.features[0][0]
        w_orig = conv_orig.weight

        conv_new = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        conv_new.weight.data = w_orig.sum(dim=1, keepdim=True)
        backbone.features[0][0] = conv_new

        for i in range(freeze_up_to): # includes stem
            for param in backbone.features[i].parameters():
                param.requires_grad = False

        self.encoder = backbone.features
        # encoder output: (B, 1280, 7, 7) for 224x224 input
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
            
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

# %%
# ── Model – EfficientNet-B0 VAE ───────────────────────────────────────────────
#
# Encoder: EfficientNet-B0 backbone.features → (B, 1280, 7, 7)
# Latent:  Two 1×1 conv heads → mu (B, latent_dim, 7, 7)
#                                log_var (B, latent_dim, 7, 7)
# Decoder: Symmetric ConvTranspose2d stack → (B, 3, 224, 224)
#
# forward() returns (recon, mu, log_var) so the training loop can compute:
#   loss = reconstruction_loss + β * KL_loss
#   KL   = -0.5 * mean(1 + log_var - mu² - exp(log_var))

class NVMELVAE_Legacy(nn.Module):
    def __init__(self, freeze_up_to: int = 0, latent_dim: int = 1024):
        """
        Args:
            freeze_up_to: freeze backbone.features[0..freeze_up_to-1] (same
                          convention as NVMELAutoencoder).
            latent_dim:   number of channels in the spatial latent map
                          (B, latent_dim, 7, 7).
        """
        super().__init__()

        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for i in range(freeze_up_to):
            for param in backbone.features[i].parameters():
                param.requires_grad = False

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = backbone.features
        # output shape: (B, 1280, 7, 7) for 224×224 input

        # ── Latent projections ────────────────────────────────────────────────
        self.fc_mu      = nn.Conv2d(1280, latent_dim, kernel_size=1)
        self.fc_log_var = nn.Conv2d(1280, latent_dim, kernel_size=1)

        # ── Decoder ───────────────────────────────────────────────────────────
        # Mirrors the NVMELAutoencoder decoder, reading from latent_dim channels
        self.decoder = nn.Sequential(
            # 7 → 14
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 14 → 28
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 28 → 56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 56 → 112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 112 → 224
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),   # output in [0, 1] — drop if inputs are normalised
        )

    # ── Reparameterisation trick ──────────────────────────────────────────────
    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        # at inference time just return the mean
        return mu

    def forward(self, x: torch.Tensor):
        """
        Returns:
            recon   – reconstructed image  (B, 3, H, W)
            mu      – latent mean          (B, latent_dim, 7, 7)
            log_var – latent log-variance  (B, latent_dim, 7, 7)
        """
        features = self.encoder(x)          # (B, 1280, 7, 7)
        mu      = self.fc_mu(features)      # (B, latent_dim, 7, 7)
        log_var = self.fc_log_var(features) # (B, latent_dim, 7, 7)
        z       = self.reparameterise(mu, log_var)
        recon   = self.decoder(z)           # (B, 3, 224, 224)
        return recon

# %%
# ── Model – EfficientNet-B0 VAE (New Interface) ─────────────────────────────
class NVMELVAE(nn.Module):
    def __init__(self, freeze_up_to: int = 0, latent_dim: int = 1024):
        """
        Args:
            freeze_up_to: freeze backbone.features[0..freeze_up_to-1]
            latent_dim:   number of channels in the spatial latent map
        """
        super().__init__()

        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for i in range(freeze_up_to):
            for param in backbone.features[i].parameters():
                param.requires_grad = False

        self.features = backbone.features
        self.fc_mu = nn.Conv2d(1280, latent_dim, kernel_size=1)
        self.fc_log_var = nn.Conv2d(1280, latent_dim, kernel_size=1)

        self.decoder = nn.Sequential(
            # 7 → 14
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 14 → 28
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 28 → 56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 56 → 112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 112 → 224
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),   # output in [0, 1] — drop if inputs are normalised
        )

    def encoder(self, x: torch.Tensor):
        features = self.features(x)
        mu = self.fc_mu(features)
        if self.training:
            log_var = self.fc_log_var(features)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))

# %%
if __name__ == "__main__":
    import os
    import glob
    import sys
    
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from utils import load_best_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_model = NVMELVAE_Legacy().to(device)
    new_model = NVMELVAE().to(device)
    
    ckpt_dir = "checkpoints/efficientnet_nv_mel_vae_legacy"
    new_ckpt_dir = "checkpoints/efficientnet_nv_mel_vae"
    os.makedirs(new_ckpt_dir, exist_ok=True)
    
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pth")))
    
    for i in range(len(ckpts)):
        print(f"Processing checkpoint {i+1}/{len(ckpts)}...")
        
        # Use utils.py function to iteratively load models
        ckpt = load_best_model(old_model, ckpt_dir, lambda c, idx=i: idx, device)
        
        # Map old state dict to new state dict keys
        old_sd = old_model.state_dict()
        new_sd = new_model.state_dict()
        
        for k, v in old_sd.items():
            if k.startswith("encoder."):
                new_k = k.replace("encoder.", "features.", 1)
            else:
                new_k = k
                
            if new_k in new_sd:
                new_sd[new_k].copy_(v)
            else:
                print(f"Warning: Key {new_k} not found in new model!")
                
        # Update checkpoint model_state and serialize
        ckpt["model_state"] = new_model.state_dict()
        
        filename = os.path.basename(ckpts[i])
        new_ckpt_path = os.path.join(new_ckpt_dir, filename)
        torch.save(ckpt, new_ckpt_path)
        print(f"Saved {new_ckpt_path}")


# %%
