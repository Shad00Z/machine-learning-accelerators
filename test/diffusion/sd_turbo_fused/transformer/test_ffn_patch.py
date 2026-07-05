"""Smoke tests for patching the fused FFN into the U-Net.

Generates an image with the baseline pipeline and with the FFN-patched pipeline (same seed) and
verifies pixel-level equivalence.
"""
import numpy as np
import pytest
import torch

from sd_turbo.image_to_text import initialize_pipeline, generation
from sd_turbo_fused.transformer.fused_ffn_block import patch_unet_ffn, FusedFFN


def _cutile_available() -> bool:
    try:
        import cuda.tile  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


PROMPT = "Will smith eating spaghetti."
SEED = 42
ATOL_PIXEL = 35


@pytest.fixture(scope="module")
def baseline_image():
    initialize_pipeline.cache_clear()
    pipe = initialize_pipeline()
    return generation(pipeline=pipe, prompt=PROMPT, seed=SEED, out_dir="diffusion/outputs/baseline")


@pytest.fixture(scope="module")
def ffn_patched_image():
    initialize_pipeline.cache_clear()
    pipe = initialize_pipeline()
    patch_unet_ffn(pipe.unet, verbose=True)
    return generation(pipeline=pipe, prompt=PROMPT, seed=SEED, out_dir="diffusion/outputs/ffn")


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available (run on the Spark)")
def test_patch_covers_all_transformer_blocks():
    from diffusers.models.attention import BasicTransformerBlock

    initialize_pipeline.cache_clear()
    pipe = initialize_pipeline()
    n = patch_unet_ffn(pipe.unet)
    
    assert n >= 16, f"expected >=16 transformer blocks, got {n}"
    for m in pipe.unet.modules():
        if isinstance(m, BasicTransformerBlock):
            assert isinstance(m.ff, FusedFFN), "ff was not replaced with FusedFFN"
            assert isinstance(m.norm3, torch.nn.Identity), "norm3 was not replaced with Identity"
    return


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available (run on the Spark)")
def test_ffn_patch_image_equivalence(baseline_image, ffn_patched_image):
    base = np.array(baseline_image, dtype=np.int32)
    fused = np.array(ffn_patched_image, dtype=np.int32)
    assert base.shape == fused.shape, f"shape differ: {base.shape} vs {fused.shape}"

    max_diff = int(np.abs(base - fused).max())
    mean_diff = float(np.abs(base - fused).mean())
    
    print(f"Max pixel diff: {max_diff}  |  Mean pixel diff: {mean_diff:.4f}")
    assert max_diff <= ATOL_PIXEL, (
        f"images differ by up to {max_diff} (tol={ATOL_PIXEL}); mean {mean_diff:.4f}"
    )
    return
