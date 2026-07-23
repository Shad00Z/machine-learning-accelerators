"""Benchmark: unfused (LayerNorm + GEGLU) vs fused cuTile kernel.
"""

import math
import torch

from sd_turbo.transformer.reference import ln_geglu_reference
from sd_turbo.transformer.kernels.ln_geglu import launch_ln_geglu_mm1
from utils.helper import _cutile_available

WARMUP = 50
ITERS  = 200

T, K, N1 = 77, 320, 2560
EPS = 1e-5


def _timed(fn, *args, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Return median latency in milliseconds over multiple measured runs."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def main():
    if not _cutile_available():
        print("CUDA not available – skipping benchmark.")
        return

    torch.manual_seed(0)
    x     = torch.randn(T,  K,  dtype=torch.float16, device="cuda")
    ln_w  = torch.randn(K,      dtype=torch.float16, device="cuda") * 0.1 + 1.0
    ln_b  = torch.randn(K,      dtype=torch.float16, device="cuda") * 0.1
    W1    = torch.randn(N1, K,  dtype=torch.float16, device="cuda") / math.sqrt(K)
    b1    = torch.randn(N1,     dtype=torch.float16, device="cuda") * 0.1

    print(f"Shape: T={T}, K={K}, N1={N1}  (N_half={N1//2})")
    print(f"Warmup iters: {WARMUP}  Measured iters: {ITERS}\n")

    t_unfused = _timed(ln_geglu_reference, x, ln_w, ln_b, W1, b1, EPS)
    t_fused   = _timed(launch_ln_geglu_mm1, x, ln_w, ln_b, W1, b1, EPS)

    speedup = t_unfused / t_fused if t_fused > 0 else float("inf")

    print(f"{'Variant':<20} {'Median (ms)':>12}")
    print("-" * 34)
    print(f"{'Unfused (ref)':<20} {t_unfused:>12.4f}")
    print(f"{'Fused (cuTile)':<20} {t_fused:>12.4f}")
    print("-" * 34)
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
