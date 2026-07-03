"""Benchmark: unfused (GroupNorm + SiLU) vs fused cuTile kernel.
"""

import torch

from sd_turbo.resnet.resnet_block import gn_silu_reference
from utils.helper import _cutile_available
from sd_turbo_fused.resnet.gn_silu_kernel import launch_reference_config_kernel
from sd_turbo_fused.resnet.gn_silu_split_kernel import launch_split_config_kernel
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


def bench_shape(B, C, H, W, G=32, eps=1e-5):
    x = torch.randn(B, C, H, W, dtype=torch.float16, device="cuda")
    w = torch.randn(C, dtype=torch.float16, device="cuda")
    b = torch.randn(C, dtype=torch.float16, device="cuda")
    compiled = torch.compile(gn_silu_reference)

    t_e = _timed(gn_silu_reference,              x, w, b, G, eps)
    t_c = _timed(compiled,                       x, w, b, G, eps)
    t_f = _timed(launch_reference_config_kernel, x, w, b, G, eps)
    t_s = _timed(launch_split_config_kernel,     x, w, b, G, eps)

    ws = 2 * B * C * H * W * 2 / 2**20   # x+out working set in MiB
    print(f"B={B:3d} {C}x{H}x{W}  ws={ws:6.0f}MiB | "
          f"eager {t_e*1e3:7.1f}  compile {t_c*1e3:7.1f}  "
          f"cuTile {t_f*1e3:7.1f}  split {t_s*1e3:7.1f} us "
          f"| cuTile/compile {t_c/t_f:.2f}x  split/cuTile {t_f/t_s:.2f}x")


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

    t_eager       = _timed(gn_silu_reference,              x, w, b, ng, eps)
    t_compiled    = _timed(gn_silu_compiled,               x, w, b, ng, eps)
    t_fused       = _timed(launch_reference_config_kernel, x, w, b, ng, eps)
    t_fused_split = _timed(launch_split_config_kernel,     x, w, b, ng, eps)

    header = (f"{'Variant':<22} {'Median (ms)':>12} {'vs eager':>10} "
              f"{'vs compile':>12} {'vs fused':>10}")
    print(header)
    print("-" * len(header))
    for label, t in (("Unfused (eager)",      t_eager),
                     ("torch.compile",        t_compiled),
                     ("Fused (cuTile)",       t_fused),
                     ("Fused Split (cuTile)", t_fused_split)):
        vs_eager   = t_eager    / t if t > 0 else float("inf")
        vs_compile = t_compiled / t if t > 0 else float("inf")
        vs_fused   = t_fused    / t if t > 0 else float("inf")
        print(f"{label:<22} {t:>12.4f} {vs_eager:>9.2f}x {vs_compile:>11.2f}x {vs_fused:>9.2f}x")
    print("-" * len(header))
    
    # for B in (1, 2, 4, 8, 16, 32, 64):
    #     x = torch.randn(B, 320, 64, 64, dtype=torch.float16, device="cuda")
    #     w = torch.randn(320, dtype=torch.float16, device="cuda")
    #     b = torch.randn(320, dtype=torch.float16, device="cuda")
    #     t = _timed(launch_reference_config_kernel, x, w, b, 32, 1e-5)   # batched _timed
    #     print(f"B={B:3d}  {t*1e3:7.2f} us/call  {t/B*1e3:6.2f} us/image")
    
    # For larger shapes (crossing the 24 MiB L2 boundary into DRAM)
    for B in (1, 8, 32, 64, 128):
        bench_shape(B, 320, 64, 64)


if __name__ == "__main__":
    main()
