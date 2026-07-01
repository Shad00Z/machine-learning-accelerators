"""Benchmark: unfused (GroupNorm + SiLU) vs fused cuTile kernel.
"""

import torch

from sd_turbo.resnet.resnet_block import gn_silu_reference
from utils.helper import _cutile_available
from sd_turbo_fused.resnet.gn_silu_kernel import launch_reference_config_kernel
from data.load_helper import data_path, load_data

WARMUP = 50
ITERS = 200


def _timed(fn, *args, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Return median latency in milliseconds over multiple measured runs."""
    # warmup
    for _ in range(warmup):
        
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median


def _load_ref_cuda():
    path = data_path()
    if not path.exists():
        raise FileNotFoundError(f"Reference data not found: {path}")
    data = load_data(path)
    # Move every tensor to CUDA
    return {
        k: v.cuda() if isinstance(v, torch.Tensor) else v
        for k, v in data.items()
    }


def main():
    if not _cutile_available():
        print("CUDA not available – skipping benchmark.")
        return

    print("Loading reference data")
    ref = _load_ref_cuda()
    x, w, b = ref["x"], ref["weight"], ref["bias"]
    ng, eps = ref["num_groups"], ref["eps"]

    print(f"Input shape : {x.shape}  dtype={x.dtype}  device={x.device}")
    print(f"Groups      : {ng}  eps={eps}")
    print(f"Warmup iters: {WARMUP}  Measured iters: {ITERS}\n")

    t_unfused = _timed(gn_silu_reference, x, w, b, ng, eps)
    t_fused   = _timed(launch_reference_config_kernel, x, w, b, ng, eps)

    speedup = t_unfused / t_fused if t_fused > 0 else float("inf")

    print(f"{'Variant':<20} {'Median (ms)':>12}")
    print("-" * 34)
    print(f"{'Unfused (ref)':<20} {t_unfused:>12.4f}")
    print(f"{'Fused (cuTile)':<20} {t_fused:>12.4f}")
    print("-" * 34)
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
