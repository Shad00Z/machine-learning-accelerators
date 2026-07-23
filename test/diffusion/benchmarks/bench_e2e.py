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

from sd_turbo.resnet.fused_block import patch_unet
from sd_turbo.transformer.fused_block import patch_unet_ffn
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


def _run_pipeline(pipe, batch_size: int = 1, seed: int = SEED) -> None:
    generator = torch.Generator("cuda").manual_seed(seed)
    pipe(
        prompt=PROMPT,
        num_images_per_prompt=batch_size,
        num_inference_steps=1,
        guidance_scale=0.0,
        generator=generator,
        output_type="latent",
    )


def _timed(pipe, batch_size: int = 1, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Median end-to-end latency in milliseconds (for a whole batch)."""
    for _ in range(warmup):
        _run_pipeline(pipe, batch_size)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        _run_pipeline(pipe, batch_size)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def _patch_both(unet):
    patch_unet(unet)
    patch_unet_ffn(unet)
    return


def _variant(patch_fn=None, compile_unet=False, batch_size: int = 1) -> float:
    """Fresh pipeline, optionally patched and/or compiled, then timed at the given batch size."""
    pipe = _fresh_pipeline()
    if patch_fn is not None:
        patch_fn(pipe.unet)
    if compile_unet:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
        t = _timed(pipe, batch_size, warmup=WARMUP + 2)   # extra warmup to absorb compilation
    else:
        t = _timed(pipe, batch_size)
    del pipe
    return t


BATCH_SIZES = (1, 4, 8, 16)   # e2e sweep; extend as memory allows (512^2 images)


def batch_sweep():
    """End-to-end latency vs batch size -- does fusion beat torch.compile once GN is DRAM-bound?

    The GN profitability gate was tuned at batch 1; at batch >= 8 the working set spills to DRAM and
    the fused kernel wins on *all* shapes, so we open the gate (fuse every GN shape) to give the fusion
    its best shot. The FFN gate stays (dim=320 only) -- the FFN is compute-bound at any batch, so fusing
    its other shapes would only regress. Reported per whole batch (not per image).
    """
    import sd_turbo.resnet.fused_block as gnmod
    saved_gate = gnmod._VALID_GN_SHAPES
    gnmod._VALID_GN_SHAPES = None   # fuse every correctness-eligible GN shape

    configs = [
        ("Baseline",      None,        False),
        ("Fused ResNet",  patch_unet,  False),
        ("Fused Both",    _patch_both, False),
        ("torch.compile", None,        True),
    ]
    try:
        print("\n===== End-to-end batch sweep (GN gate OPEN: all shapes fused) =====")
        print(f"{'Batch':>5} {'Baseline':>10} {'F-ResNet':>10} {'F-Both':>10} {'compile':>10}"
              f" {'Both/base':>11} {'Both/comp':>11} {'ResNet/comp':>12}")
        for B in BATCH_SIZES:
            r = {label: _variant(patch_fn, comp, batch_size=B) for label, patch_fn, comp in configs}
            print(f"{B:>5} {r['Baseline']:>10.1f} {r['Fused ResNet']:>10.1f} {r['Fused Both']:>10.1f}"
                  f" {r['torch.compile']:>10.1f}"
                  f" {r['Baseline']/r['Fused Both']:>10.2f}x {r['torch.compile']/r['Fused Both']:>10.2f}x"
                  f" {r['torch.compile']/r['Fused ResNet']:>11.2f}x")
    finally:
        gnmod._VALID_GN_SHAPES = saved_gate   # restore production gate
        return


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
    
    batch_sweep()
    return


if __name__ == "__main__":
    main()
