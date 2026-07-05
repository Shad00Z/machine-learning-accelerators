"""Benchmark: end-to-end SD-Turbo image generation across fusion configurations.

Compares five variants at the whole-pipeline level:
  1. Baseline       - unmodified pipeline (eager)
  2. Fused ResNet   - GroupNorm+SiLU replaced by the cuTile kernel (patch_unet)
  3. Fused FFN      - transformer LayerNorm+GEGLU replaced by the cuTile kernel (patch_unet_ffn)
  4. Fused Both     - ResNet + FFN fused
  5. torch.compile  - baseline UNet compiled (mode='reduce-overhead', i.e. CUDA graphs)

Speedups are reported vs baseline and vs torch.compile.
"""
import torch
from diffusers import AutoPipelineForText2Image

from sd_turbo_fused.resnet.fused_resnet_block import patch_unet
from sd_turbo_fused.transformer.fused_ffn_block import patch_unet_ffn
from utils.helper import _cutile_available

PROMPT = "A photo of a cat wearing a hat, sitting on a bench in the park, with a sunny background, highly detailed."
SEED = 42
WARMUP = 10
ITERS = 40


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
    """Median end-to-end latency in milliseconds."""
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


def _patch_both(unet):
    patch_unet(unet)
    patch_unet_ffn(unet)
    return


def _variant(patch_fn=None, compile_unet=False) -> float:
    """Fresh pipeline, optionally patched and/or compiled, then timed."""
    pipe = _fresh_pipeline()
    if patch_fn is not None:
        patch_fn(pipe.unet)
    if compile_unet:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
        t = _timed(pipe, warmup=WARMUP + 2)   # extra warmup to absorb compilation
    else:
        t = _timed(pipe)
    del pipe
    return t


def main():
    if not _cutile_available():
        print("CUDA not available - skipping benchmark.")
        return

    print(f"Prompt      : {PROMPT!r}")
    print(f"Warmup iters: {WARMUP}  Measured iters: {ITERS}\n")

    configs = [
        ("Baseline",       None,           False),
        ("Fused ResNet",   patch_unet,     False),
        ("Fused FFN",      patch_unet_ffn, False),
        ("Fused Both",     _patch_both,    False),
        ("torch.compile",  None,           True),
    ]

    results = []
    for label, patch_fn, comp in configs:
        print(f"===== {label} =====")
        results.append((label, _variant(patch_fn, comp)))

    t_base = dict(results)["Baseline"]
    t_comp = dict(results)["torch.compile"]

    print(f"\n{'Variant':<16} {'Median (ms)':>12} {'vs baseline':>13} {'vs compile':>12}")
    print("-" * 55)
    for label, t in results:
        print(f"{label:<16} {t:>12.2f} {t_base / t:>12.2f}x {t_comp / t:>11.2f}x")
    print("-" * 55)


if __name__ == "__main__":
    main()
