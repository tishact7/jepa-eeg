import json, os, random, platform, time
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_generator(seed):
    """CPU generator used only for mask sampling, so masks are reproducible per seed."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def sample_masks(batch_size, generator, n_patches=10, n_context=6, n_target=2):
    """Per-sample disjoint context/target patch indices.
    Context = 6 random patches; target = 2 random patches from the other 4.
    Returns ctx_idx (B,6), tgt_idx (B,2), both long."""
    perm = torch.rand(batch_size, n_patches, generator=generator).argsort(dim=1)
    return perm[:, :n_context], perm[:, n_context:n_context + n_target]


def gather_patches(patches, idx):
    """patches (B,10,22,50), idx (B,k) -> (B,k,22,50)."""
    return torch.gather(patches, 1, idx[:, :, None, None].expand(-1, -1, *patches.shape[2:]))


def save_checkpoint(path, epoch, models, optimizer, loss, extra=None):
    """models: dict name -> nn.Module."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "models": {k: m.state_dict() for k, m in models.items()},
        "optimizer": optimizer.state_dict(),
        "extra": extra or {},
    }, path)


def load_checkpoint(path, models, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    for k, m in models.items():
        m.load_state_dict(ckpt["models"][k])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["loss"], ckpt["extra"]


class JsonlLogger:
    """Appends one JSON object per line. Never overwrites, so failures are kept."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path

    def log(self, **kw):
        kw["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a") as f:
            f.write(json.dumps(kw) + "\n")


def env_info():
    import moabb, mne
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "mne": mne.__version__,
        "moabb": moabb.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
