# %%
import torch

# %%
def load_best_model(model, checkpoint_dir, key, selector, device, pattern="epoch_*.pth"):
    """Given a directory of training checkpoints, load the best one based on given key (like "val_loss") and selector (like min)"""

    existing = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
    assert len(existing) > 0, "No checkpoint found"

    if len(existing) > 1:
        latest = existing[-1]
        ckpt   = torch.load(latest, map_location="cpu", weights_only=False)
        best_epoch_idx = ckpt[key].index(selector(ckpt[key]))
    else:
        best_epoch_idx = 0

    best_ckpt = torch.load(existing[best_epoch_idx], map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])

    return best_ckpt

# %%
