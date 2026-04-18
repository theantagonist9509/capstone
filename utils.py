# %%
import os
import glob
import torch

# %%
def load_best_model(model, checkpoint_dir, index_selector, device, pattern="epoch_*.pth"):
    existing = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
    assert len(existing) > 0, "No checkpoint found"

    if len(existing) > 1:
        latest = existing[-1]
        ckpt   = torch.load(latest, map_location="cpu", weights_only=False)
        best_epoch_idx = index_selector(ckpt)
    else:
        best_epoch_idx = 0

    best_ckpt = torch.load(existing[best_epoch_idx], map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])

    return best_ckpt

# %%
def print_checkpoint_info(ckpt):
    for k, v in ckpt.items():
        if "state" in k:
            continue

        if isinstance(v, dict):
            print_checkpoint_info(v) # nested history variables
            continue

        if isinstance(v, list):
            v = v[-1] # history variables
        
        print(f"{k}: {v}")

# %%
def get_orthogonal_pca_bases(embeddings: torch.Tensor, reference: torch.Tensor, k: int = 2) -> torch.Tensor:
    """
    Computes PCA on the embeddings' components orthogonal to the reference.
    
    Args:
        embeddings: Tensor of shape (B, ...) where dim 0 is batch size.
        reference: Tensor of shape (...) matching a single embedding.
        k: Number of bases to return.
        
    Returns:
        bases: Tensor of shape (k, ...) containing the PCA bases.
    """
    orig_shape = reference.shape
    B = embeddings.shape[0]
    
    X = embeddings.view(B, -1)
    r = reference.view(-1)
    
    r_norm = r / (r.norm(p=2) + 1e-8)
    
    projs = torch.matmul(X, r_norm).unsqueeze(1) * r_norm.unsqueeze(0)
    X_ortho = X - projs
    
    X_centered = X_ortho - X_ortho.mean(dim=0, keepdim=True)
    
    _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    
    bases = Vh[:k]
    
    return bases.view(k, *orig_shape)

# %%
