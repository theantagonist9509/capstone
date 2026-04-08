# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pytorch_msssim import MS_SSIM

# %%
class VGGLoss(nn.Module):
    def __init__(self, probes=(3, 8, 17), weights=(1, 1, 1)):
        super().__init__()
        assert len(probes) == len(weights), "probes and weights must match in length"

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.net = nn.Sequential(*list(vgg)[:max(probes) + 1])
        self.probes = probes
        self.weights = weights

        for p in self.net.parameters():
            p.requires_grad = False

        self._activations = []
        for probe in probes:
            self.net[probe].register_forward_hook(self._make_hook())

    def _make_hook(self):
        def hook(module, input, output):
            self._activations.append(output)
        return hook

    def forward(self, a, b):
        self._activations.clear()
        self.net(a)
        self.net(b)

        n = len(self.probes)
        loss = torch.tensor(0.0, device=a.device)
        for a_act, b_act, w in zip(self._activations[:n], self._activations[n:], self.weights):
            loss = loss + w * F.mse_loss(a_act, b_act)

        return loss

class MS_SSIMLoss(nn.Module):
    def __init__(self, channels, denorm_mean, denorm_std):
        super().__init__()
        self.ms_ssim = MS_SSIM(data_range=1, size_average=True, channel=channels)
        self.denorm_mean = denorm_mean.view(1, channels, 1, 1)
        self.denorm_std = denorm_std.view(1, channels, 1, 1)
    
    def forward(self, a, b):
        denorm_a = a * self.denorm_std + self.denorm_mean
        denorm_b = b * self.denorm_std + self.denorm_mean

        denorm_a = torch.clamp(denorm_a, 0, 1)
        denorm_b = torch.clamp(denorm_b, 0, 1)

        return 1 - self.ms_ssim(denorm_a, denorm_b)

# %%
# Test VGGLoss
if __name__ == "__main__":
    output = torch.randn(1, 3, 224, 224, requires_grad=True)
    target = torch.randn(1, 3, 224, 224)

    vgg_loss = VGGLoss()

    loss = vgg_loss(output, target)
    print(loss)

# %%
    loss.backward()
    print(output.grad)

# %%
# Test MS_SSIMLoss
if __name__ == "__main__":
    output = torch.randn(1, 3, 224, 224, requires_grad=True)
    target = torch.randn(1, 3, 224, 224)

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    norm_output = (output - imagenet_mean) / imagenet_std
    norm_target = (target - imagenet_mean) / imagenet_std

    ms_ssim_loss = MS_SSIMLoss(channels=3, denorm_mean=imagenet_mean, denorm_std=imagenet_std)

    loss = ms_ssim_loss(norm_output, norm_target)
    print(loss)

# %%
    loss.backward()
    print(output.grad)

# %%
