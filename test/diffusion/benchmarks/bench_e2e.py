"""Benchmark: end-to-end SD-Turbo image generation.

Compares three variants:
  1. Baseline       - unmodified pipeline
  2. Fused (cuTile) - GroupNorm+SiLU replaced by the cuTile kernel
  3. torch.compile  - baseline pipeline with torch.compile on the UNet
"""

import torch
from diffusers import AutoPipelineForText2Image

from sd_turbo_fused.resnet.fused_resnet_block import patch_unet
from utils.helper import _cutile_available

PROMPT = "A photo of a cat wearing a hat, sitting on a bench in the park, with a sunny background, highly detailed."
SEED = 42
WARMUP = 3
ITERS = 10


def _fresh_pipeline():
    """Load a new, unshared pipeline instance (bypasses lru_cache)."""
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to("cuda")
    return pipe


def _run_pipeline(pipe, seed: int = SEED) -> None:
    generator = torch.Generator("cuda").manual_seed(seed)
    pipe(
        prompt=PROMPT,
        num_inference_steps=1,
        guidance_scale=0.0,
        generator=generator,
        output_type="latent",
    )


def _timed(pipe, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Return median end-to-end latency in milliseconds."""
    for _ in range(warmup):
        _run_pipeline(pipe)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(iters):
        start.record()
        _run_pipeline(pipe)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def main():
    if not _cutile_available():
        print("CUDA not available – skipping benchmark.")
        return

    print(f"Prompt      : {PROMPT!r}")
    print(f"Warmup iters: {WARMUP}  Measured iters: {ITERS}\n")

    # 1. Baseline
    print("===== Loading pipeline (baseline) =====")
    pipe = _fresh_pipeline()
    t_baseline = _timed(pipe)
    del pipe

    # 2. Fused (cuTile)
    print("\n===== Loading pipeline for fused variant =====")
    pipe_fused = _fresh_pipeline()
    n_patched = patch_unet(pipe_fused.unet, verbose=False)
    print(f"  {n_patched} ResnetBlock2D blocks patched")
    t_fused = _timed(pipe_fused)
    del pipe_fused

    # 3. torch.compile
    print("\n===== Loading pipeline for torch.compile variant =====")
    pipe_compiled = _fresh_pipeline()
    print("  Compiling UNet (mode='reduce-overhead')…")
    pipe_compiled.unet = torch.compile(pipe_compiled.unet, mode="reduce-overhead")
    # Extra warmup to absorb compilation on first real call
    t_compiled = _timed(pipe_compiled, warmup=WARMUP + 2)
    del pipe_compiled

    # Results
    print(f"\n{'Variant':<25} {'Median (ms)':>12} {'Speedup vs baseline':>20}")
    print("-" * 60)
    for label, t in [
        ("Baseline", t_baseline),
        ("Fused (cuTile)", t_fused),
        ("torch.compile", t_compiled),
    ]:
        speedup = t_baseline / t if t > 0 else float("inf")
        print(f"{label:<25} {t:>12.2f} {speedup:>19.2f}x")
    print("-" * 60)


if __name__ == "__main__":
    main()
