import numpy as np
import os
import torch

from pathlib import Path


def repo_root(marker=".git"):
    """
    Find the repository root (traverse up).
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"repo root ({marker}) not found above {__file__}")


def data_path(name="gn_silu_block0.npz"):
    base = os.environ.get("GN_SILU_REF_DIR") or (repo_root() / "test" / "diffusion" / "data")
    return Path(base, name)


def load_data(path):
    """
    Load data based on the provided path for a *.npz file.

    Returns a dict: x, weight, bias, y, num_groups, num_channels, eps
    """
    npz = np.load(path)
    return {
        "x":      torch.from_numpy(npz["x"]),
        "weight": torch.from_numpy(npz["weight"]),
        "bias":   torch.from_numpy(npz["bias"]),
        "y":      torch.from_numpy(npz["y"]),

        "num_groups":   int(npz["groups"]),
        "num_channels": int(npz["num_channels"]),
        "eps":          float(npz["eps"]),
    }
