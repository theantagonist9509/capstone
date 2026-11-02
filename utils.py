# %%
import os
import glob
import torch


# %%
def _use_kaggle_input() -> bool:
    return os.environ.get("CAPSTONE_USE_INPUT", "0").lower() in ("1", "true", "yes")


def kaggle_input_dir(required, hint=None, root="/kaggle/input"):
    """Find the mounted Kaggle dataset directory containing all `required`
    entries. `hint` optionally restricts the path to substrings (used to tell
    apart datasets with identical folder names)."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"'{root}' does not exist — is this on Kaggle?")
    for dirpath, dirnames, _ in os.walk(root):
        if dirpath.count(os.sep) > 7:
            dirnames[:] = []
            continue
        try:
            if not set(required).issubset(os.listdir(dirpath)):
                continue
        except OSError:
            continue
        if hint is None or any(h.lower() in dirpath.lower() for h in hint):
            return dirpath
    raise FileNotFoundError(
        f"no directory under '{root}' contains {sorted(required)}"
        + (f" and matches {hint}" if hint else "")
    )


def dataset_dir(local_dir, required, hint=None):
    """`local_dir` by default; with CAPSTONE_USE_INPUT=1, the mounted Kaggle
    source instead. Temporary A/B switch for the /kaggle/input question."""
    if _use_kaggle_input():
        return kaggle_input_dir(required, hint)
    return local_dir


# %%
def in_ipython() -> bool:
    """True when running under IPython / Jupyter (incl. Kaggle notebooks)."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def maybe_show(fig=None, show: bool = True) -> None:
    """
    Show a figure only when it will actually render (IPython/notebook with an
    interactive/inline backend). Under a plain script run with the Agg
    backend, close it instead so nothing pops up and memory is freed.
    """
    import matplotlib
    backend = matplotlib.get_backend().lower()
    interactive = in_ipython() or backend.startswith(("module://matplotlib_inline",
                                                      "tkagg", "qt", "nbagg", "ipympl"))
    if show and interactive:
        import matplotlib.pyplot as plt
        plt.show()
    elif not show or not interactive:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)

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
