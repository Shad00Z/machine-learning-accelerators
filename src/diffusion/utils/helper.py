import torch

def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def is_shape_fusable(C: int, H: int, W: int, num_groups: int) -> bool:
    """
    Checks if cuTile can handle the shape (GN+SiLU kernel)
    """
    return _is_pow2(H) and _is_pow2(W) and C % num_groups == 0


def _cutile_available() -> bool:
    """
    Checks if CUDA is available. Needed for pytests.
    """
    try:
        import cuda.tile  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()
