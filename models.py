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
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
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

# %%
