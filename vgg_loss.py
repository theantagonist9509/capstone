# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

# %%
if __name__ == "__main__":
    output = torch.randn(1, 3, 224, 224, requires_grad=True)
    target = torch.randn(1, 3, 224, 224)

    vgg_loss = VGGLoss()

    loss = vgg_loss(output, target)
    loss

# %%
    loss.backward()
    output.grad

# %%
