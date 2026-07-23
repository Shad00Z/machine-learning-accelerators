"""Benchmark the fused LayerNorm -> GEGLU FFN cuTile kernel (Milestone 5).

Benchmarks:
  bench_tiles        -- sweep (tM, tN, tK) for the fused kernel, one TFLOPS heatmap per k_tile.
  bench_tile_kernels -- per tile, time the three cuTile kernels (fused/swizzle/split) side by side.
  bench_compare      -- eager vs torch.compile vs cuTile variants, on the transformer FFN block.

    PYTHONPATH=src/diffusion:test/diffusion python test/diffusion/benchmarks/bench_ffn.py

Note: launch_ffn_kernel transposes w1/w2 per call (contiguous copies). In the real model these
are hoisted once, so the cuTile numbers here are slightly pessimistic.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward

from sd_turbo.transformer.kernels.ffn_gemm_split import launch_ffn_gemm_split
from sd_turbo.transformer.kernels.ffn import launch_ffn_kernel
from sd_turbo.transformer.kernels.ffn_split import launch_ffn_split
from sd_turbo.transformer.kernels.ffn_swizzle import launch_ffn_swizzle
from utils.helper import _cutile_available

WARMUP = 30
ITERS = 200

# real FFN shapes (tokens, dim) from survey_ffn_shapes; 64x1280 duplicates 256x1280's dims
SHAPES = [(4096, 320), (1024, 640), (256, 1280)]


def _timed(fn, warmup=WARMUP, iters=ITERS):
    """Median-free amortized latency in ms (launches pipelined, one sync)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


class LNFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, activation_fn="geglu")

    def forward(self, x):
        return self.ff(self.norm(x))


def _make(dim, tokens):
    m = LNFFN(dim).eval().half().cuda()
    x = torch.randn(1, tokens, dim, dtype=torch.float16, device="cuda")
    ln, proj, lin2 = m.norm, m.ff.net[0].proj, m.ff.net[-1]
    W = dict(ln_weight=ln.weight, ln_bias=ln.bias, eps=ln.eps,
             w1=proj.weight, b1=proj.bias, w2=lin2.weight, b2=lin2.bias)
    return m, x, W


def _cutile(x, W, tM=64, tN=64, tK=64):
    return launch_ffn_kernel(x, W["ln_weight"], W["ln_bias"], W["eps"],
                             W["w1"], W["b1"], W["w2"], W["b2"], tM, tN, tK)


def bench_compare(shapes=SHAPES):
    print("=== FFN latency (us); ( ) = x vs torch.compile ===")
    for tokens, dim in shapes:
        m, x, W = _make(dim, tokens)
        cm = torch.compile(m)
        args = (x, W["ln_weight"], W["ln_bias"], W["eps"], W["w1"], W["b1"], W["w2"], W["b2"])
        with torch.no_grad():
            t_e = _timed(lambda: m(x))
            t_c = _timed(lambda: cm(x))
            t_f = _timed(lambda: _cutile(x, W))
            t_w = _timed(lambda: launch_ffn_swizzle(*args))
            t_s = _timed(lambda: launch_ffn_split(*args))
            t_g = _timed(lambda: launch_ffn_gemm_split(*args))
        print(f"tokens={tokens:5d} dim={dim:5d} | eager {t_e*1e3:6.0f}  compile {t_c*1e3:6.0f}")
        for name, t in (("fused ", t_f), ("swizzle", t_w),
                        ("split ", t_s), ("gsplit", t_g)):
            print(f"    {name:>8} {t*1e3:7.1f} us  ({t_c/t:.2f}x)")


def _save_heatmap(values, tMs, tNs, tK, tokens, dim, out):
    """One m_tile x n_tile TFLOPS heatmap for a fixed k_tile (assignment style)."""
    _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(values, cmap="plasma")
    ax.set_xticks(range(len(tNs)))
    ax.set_yticks(range(len(tMs)))
    ax.set_xticklabels(tNs)
    ax.set_yticklabels(tMs)
    ax.set_xlabel("n_tile")
    ax.set_ylabel("m_tile")
    ax.set_title(f"FFN Tile Shape Throughput (TFLOPS)\ntokens={tokens}, dim={dim}, k_tile={tK} fixed")

    vmax = np.nanmax(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center",
                        color="white" if values[i, j] < vmax * 0.7 else "black",
                        fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="TFLOPS")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"  wrote {out}")


def bench_tiles(tokens=4096, dim=320):
    """One TFLOPS heatmap per valid k_tile (tK is a 3rd axis -> one PNG each)."""
    print(f"\n=== FFN tile heatmaps  tokens={tokens} dim={dim} ===")
    m, x, W = _make(dim, tokens)
    M = tokens
    inner = W["w1"].shape[0] // 2
    flop = 6 * M * inner * dim        # mm1 (4*M*dim*inner) + mm2 (2*M*inner*dim); LN/GEGLU negligible

    pow2 = (16, 32, 64, 128)
    tMs = [t for t in pow2 if tokens % t == 0]                     # m_tile (rows - no output pad)
    tNs = [t for t in pow2 if inner % t == 0 and dim % t == 0]     # n_tile (cols - no output pad)
    tKs = list(pow2)

    for tK in tKs:
        print(f"-- k_tile={tK} --")
        values = np.full((len(tMs), len(tNs)), np.nan)
        for i, tM in enumerate(tMs):
            for j, tN in enumerate(tNs):
                try:
                    t = _timed(lambda: _cutile(x, W, tM, tN, tK))     # ms/call
                    values[i, j] = flop / (t * 1e-3) / 1e12           # TFLOPS
                    print(f"  tM={tM:3d} tN={tN:3d} tK={tK:3d}  {values[i, j]:6.1f} TFLOPS  ({t*1e3:.1f} us)")
                except Exception as ex:
                    print(f"  tM={tM:3d} tN={tN:3d} tK={tK:3d}  FAILED: {type(ex).__name__}")
        _save_heatmap(values, tMs, tNs, tK, tokens, dim, out=f"ffn_tile_throughput_dim{dim}_tK{tK}.png")


def bench_tile_kernels(shapes=SHAPES, kernels=None):
    """Per tile shape, compare the cuTile FFN kernels side by side in mm1 TFLOPS."""
    if kernels is None:
        kernels = {
            "fused":   launch_ffn_kernel,   # 2-accumulator mm1
            "swizzle": launch_ffn_swizzle,  # L2 block swizzle
            "split":   launch_ffn_split,    # register-split mm1
            # "gsplit": launch_ffn_gemm_split,  # GEGLU fused into cuTile mm2 (different mm2)
        }
    names = list(kernels)
    for tokens, dim in shapes:
        m, x, W = _make(dim, tokens)
        inner = W["w1"].shape[0] // 2
        flop = 4 * tokens * inner * dim        # mm1 only (B=1 so M=tokens); matches bench_tiles
        args = (x, W["ln_weight"], W["ln_bias"], W["eps"], W["w1"], W["b1"], W["w2"], W["b2"])

        pow2 = (16, 32, 64, 128)
        tMs = [t for t in pow2 if tokens % t == 0]                     # rows
        tNs = [t for t in pow2 if inner % t == 0 and dim % t == 0]     # cols
        tKs = list(pow2)

        print(f"\n=== FFN tile kernels  tokens={tokens} dim={dim}  (mm1 TFLOPS; higher = better; * = fastest per row) ===")
        print(f"{'tM':>4} {'tN':>4} {'tK':>4} | " + " ".join(f"{n:>9}" for n in names))
        best = (float("-inf"), None, None)   # (TFLOPS, kernel, tile)
        for tK in tKs:
            for tM in tMs:
                for tN in tNs:
                    tflops = {}
                    for n, fn in kernels.items():
                        try:
                            ms = _timed(lambda fn=fn, tM=tM, tN=tN, tK=tK: fn(*args, tM, tN, tK))
                            tflops[n] = flop / (ms * 1e-3) / 1e12
                        except Exception:
                            tflops[n] = float("nan")
                    row_best = max((v for v in tflops.values() if v == v), default=float("nan"))
                    for n in names:
                        v = tflops[n]
                        if v == v and v > best[0]:
                            best = (v, n, (tM, tN, tK))
                    cells = " ".join(
                        (f"{tflops[n]:8.1f}*" if tflops[n] == row_best else f"{tflops[n]:8.1f} ")
                        if tflops[n] == tflops[n] else f"{'FAIL':>8} "
                        for n in names
                    )
                    print(f"{tM:>4} {tN:>4} {tK:>4} | {cells}")
        if best[1] is not None:
            bt = "x".join(map(str, best[2]))
            print(f"  best: {best[1]:>8}  tile {bt}  {best[0]:.1f} TFLOPS")


def main():
    if not _cutile_available():
        print("CUDA not available -- skipping benchmark.")
        return
    bench_tile_kernels()
    # bench_compare()
    # for tokens, dim in SHAPES:
    #     bench_tiles(tokens, dim)


if __name__ == "__main__":
    main()
