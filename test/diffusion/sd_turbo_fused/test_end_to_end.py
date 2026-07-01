"""
End-to-end image equivalence test.

Generates an image with the baseline SD-Turbo pipeline and with the fused
(patched) pipeline using the same seed, then verifies pixel-level equivalence.
"""

import numpy as np
import pytest
import torch

from sd_turbo.image_to_text import initialize_pipeline, generation
from sd_turbo_fused.resnet.fused_resnet_block import patch_unet
from utils.helper import _cutile_available


PROMPT = "Two male students working on a laptop."
SEED = 42
ATOL_PIXEL = 21  # max absolute difference in uint8 pixel values


@pytest.fixture(scope="module")
def baseline_image():
    """Generate a reference image with the unpatched pipeline."""
    pipe = initialize_pipeline()
    return generation(pipeline=pipe, prompt=PROMPT, seed=SEED, out_dir="diffusion/outputs/baseline")


@pytest.fixture(scope="module")
def fused_image():
    """Generate an image after patching all ResnetBlock2D blocks."""
    pipe = initialize_pipeline()
    _ = patch_unet(pipe.unet, verbose=True)
    return generation(pipeline=pipe, prompt=PROMPT, seed=SEED, out_dir="diffusion/outputs/fused")


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available")
def test_end_to_end_image_equivalence(baseline_image, fused_image):
    """Pixel values of baseline and fused images must be identical (or within
    a tiny tolerance caused by fp16 rounding on the non-320ch blocks)."""
    base = np.array(baseline_image, dtype=np.int32)
    fused = np.array(fused_image, dtype=np.int32)

    assert base.shape == fused.shape, (
        f"Image shapes differ: baseline={base.shape}, fused={fused.shape}"
    )

    max_diff = int(np.abs(base - fused).max())
    mean_diff = float(np.abs(base - fused).mean())
    print(f"Max pixel diff: {max_diff}  |  Mean pixel diff: {mean_diff:.4f}")

    assert max_diff <= ATOL_PIXEL, (
        f"Images differ by up to {max_diff} pixel values (tolerance={ATOL_PIXEL}).\n"
        f"Mean absolute difference: {mean_diff:.4f}"
    )


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available")
def test_patch_unet_covers_all_resnet_blocks():
    """Smoke-test: patch_unet must touch at least 22 blocks (as documented)."""
    from diffusers.models.resnet import ResnetBlock2D
    from sd_turbo_fused.resnet.fused_resnet_block import GnSiluFused

    pipe = initialize_pipeline()
    n = patch_unet(pipe.unet)

    assert n >= 22, f"Expected >=22 patched blocks, got {n}"

    for module in pipe.unet.modules():
        if isinstance(module, ResnetBlock2D):
            assert isinstance(module.norm1, GnSiluFused), (
                "norm1 was not replaced with GnSiluFused"
            )
            assert isinstance(module.norm2, GnSiluFused), (
                "norm2 was not replaced with GnSiluFused"
            )
