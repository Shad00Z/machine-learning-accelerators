"""Benchmark: unfused (GroupNorm + SiLU) vs fused cuTile kernel.
"""

import torch

from sd_turbo.resnet.reference import gn_silu_reference
from utils.helper import _cutile_available
from sd_turbo.resnet.kernels.gn_silu import launch_reference_config_kernel
from sd_turbo.resnet.kernels.gn_silu_split import launch_split_config_kernel
from sd_turbo.resnet.kernels.gn_silu_single import launch_single_pass_kernel
from data.load_helper import data_path, load_data

WARMUP = 50
ITERS = 200


def _timed(fn, *args, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):          # NO per-iter sync -- let them pipeline
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters   # ms per call, amortized


# Alternative calculation
# def _timed(fn, *args, warmup: int = WARMUP, iters: int = ITERS) -> float:
#     """Return median latency in milliseconds over multiple measured runs."""
#     # warmup
#     for _ in range(warmup):
        
#         fn(*args)
#     torch.cuda.synchronize()

#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)

#     times = []
#     for _ in range(iters):
#         start.record()
#         fn(*args)
#         end.record()
#         torch.cuda.synchronize()
#         times.append(start.elapsed_time(end))

#     times.sort()
#     return times[len(times) // 2]  # median


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


def _bench_shape(B, C, H, W, G=32, eps=1e-5):
    """Time all variants for one (B, C, H, W); returns us per call (+ working set)."""
    x = torch.randn(B, C, H, W, dtype=torch.float16, device="cuda")
    w = torch.randn(C, dtype=torch.float16, device="cuda")
    b = torch.randn(C, dtype=torch.float16, device="cuda")
    compiled = torch.compile(gn_silu_reference)
    us = lambda fn: _timed(fn, x, w, b, G, eps) * 1e3   # ms -> us
    return {
        "eager":   us(gn_silu_reference),
        "compile": us(compiled),
        "cuTile":  us(launch_reference_config_kernel),   # 2-kernel reference
        "split":   us(launch_split_config_kernel),       # split-reduction
        "single":  us(launch_single_pass_kernel),        # single-pass
        "ws":      2 * B * C * H * W * 2 / 2**20,         # x + out, MiB
    }


def bench_table(configs, title):
    """Aligned table: eager | compile | all 3 cuTile kernels (us) + ratios + gate, per (B, C, H, W)."""
    hdr = (f"{'batch':>5} {'shape':>11} {'WS/MiB':>7} {'eager':>7} {'compile':>8} "
           f"{'cuTile':>7} {'split':>7} {'single':>7} {'cuT/cmp':>8} {'sgl/cmp':>8} {'gate':>7}")
    print(f"\n=== {title} ===")
    print(hdr)
    print("-" * len(hdr))
    for B, C, H, W in configs:
        r = _bench_shape(B, C, H, W)
        gate = "fuse" if r["cuTile"] < r["eager"] else "eager"   # profitable = beats eager
        print(f"{B:>5} {f'{C}x{H}x{W}':>11} {r['ws']:>7.0f} {r['eager']:>7.1f} {r['compile']:>8.1f} "
              f"{r['cuTile']:>7.1f} {r['split']:>7.1f} {r['single']:>7.1f} "
              f"{r['compile']/r['cuTile']:>7.2f}x {r['compile']/r['single']:>7.2f}x {gate:>7}")


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
    
    # torch.compile baseline
    gn_silu_compiled = torch.compile(gn_silu_reference)

    t_eager        = _timed(gn_silu_reference,              x, w, b, ng, eps)
    t_compiled     = _timed(gn_silu_compiled,               x, w, b, ng, eps)
    t_fused        = _timed(launch_reference_config_kernel, x, w, b, ng, eps)
    t_fused_split  = _timed(launch_split_config_kernel,     x, w, b, ng, eps)
    t_fused_single = _timed(launch_single_pass_kernel,      x, w, b, ng, eps)

    header = (f"{'Variant':<22} {'Median (ms)':>12} {'vs eager':>10} "
              f"{'vs compile':>12} {'vs fused':>10}")
    print(header)
    print("-" * len(header))
    for label, t in (("Unfused (eager)",       t_eager),
                     ("torch.compile",         t_compiled),
                     ("Fused (cuTile)",        t_fused),
                     ("Fused Split (cuTile)",  t_fused_split),
                     ("Fused Single (cuTile)", t_fused_single)):
        vs_eager   = t_eager    / t if t > 0 else float("inf")
        vs_compile = t_compiled / t if t > 0 else float("inf")
        vs_fused   = t_fused    / t if t > 0 else float("inf")
        print(f"{label:<22} {t:>12.4f} {vs_eager:>9.2f}x {vs_compile:>11.2f}x {vs_fused:>9.2f}x")
    print("-" * len(header))
    
    print("-" * len(header))
    for B in (1, 2, 4, 8, 16, 32, 64):
        x = torch.randn(B, 320, 64, 64, dtype=torch.float16, device="cuda")
        w = torch.randn(320, dtype=torch.float16, device="cuda")
        b = torch.randn(320, dtype=torch.float16, device="cuda")
        t = _timed(launch_reference_config_kernel, x, w, b, 32, 1e-5)   # batched _timed
        print(f"B={B:3d}  {t*1e3:7.2f} us/call  {t/B*1e3:6.2f} us/image")
    print("-" * len(header))
    
    # Real U-Net GroupNorm shapes at batch 1 (sorted by spatial size H*W, matches the report table)
    real_shapes = [(320, 64, 64), (640, 64, 64), (960, 64, 64),
                   (320, 32, 32), (640, 32, 32), (960, 32, 32), (1280, 32, 32), (1920, 32, 32),
                   (640, 16, 16), (1280, 16, 16), (1920, 16, 16), (2560, 16, 16),
                   (1280, 8, 8), (2560, 8, 8)]
    bench_table([(1, C, H, W) for (C, H, W) in real_shapes],
                "Real U-Net GroupNorm shapes (batch 1)")

    # Batch sweep on 320x64x64 (crossing the 24 MiB L2 boundary into DRAM)
    bench_table([(B, 320, 64, 64) for B in (1, 8, 32, 64, 128)],
                "Batch sweep - 320x64x64")


if __name__ == "__main__":
    main()
