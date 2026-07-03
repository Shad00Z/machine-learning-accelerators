import pytest, torch

from sd_turbo.image_to_text import initialize_pipeline, survey_groupnorm_shapes
from sd_turbo.resnet.resnet_block import gn_silu_reference
from sd_turbo_fused.resnet.gn_silu_kernel import launch_reference_config_kernel
from sd_turbo_fused.resnet.gn_silu_split_kernel import launch_split_config_kernel
from utils.helper import _cutile_available

ATOL, RTOL, EPS = 2e-2, 2e-2, 1e-5

# distinct (C, H, W) from the survey; groups is 32 everywhere
SHAPES = [(320,64,64),(640,32,32),(1280,16,16),(1280,8,8),(2560,8,8),(640,64,64),
          (2560,16,16),(320,32,32),(640,16,16),(960,32,32),(960,64,64),(1280,32,32),
          (1920,16,16),(1920,32,32)]

@pytest.mark.parametrize("C,H,W", SHAPES)
@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available")
def test_kernel_matches_reference_shape(C, H, W):
    torch.manual_seed(0)
    G, eps = 32, 1e-5
    x      = torch.randn(1, C, H, W, dtype=torch.float16, device="cuda")
    weight = torch.randn(C, dtype=torch.float16, device="cuda")
    bias   = torch.randn(C, dtype=torch.float16, device="cuda")
    out = launch_reference_config_kernel(x, weight, bias, G, eps)
    ref = gn_silu_reference(x, weight, bias, G, eps)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    return


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available")
def test_kernel_supports_all_model_shapes():
    """
    Every GroupNorm shape the real U-Net uses must match the torch reference.
    """
    # discover real shapes
    counts = survey_groupnorm_shapes(initialize_pipeline())
    torch.manual_seed(0)
    
    for (C, H, W, G) in counts:
        x      = torch.randn(1, C, H, W, dtype=torch.float16, device="cuda")
        weight = torch.randn(C, dtype=torch.float16, device="cuda")
        bias   = torch.randn(C, dtype=torch.float16, device="cuda")
        out = launch_reference_config_kernel(x, weight, bias, G, EPS)
        ref = gn_silu_reference(x, weight, bias, G, EPS)
        torch.testing.assert_close(
            out, ref, atol=ATOL, rtol=RTOL,
            msg=lambda m: f"shape (C={C}, H={H}, W={W}, G={G}) mismatch:\n{m}",
        )
    return


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available")
def test_split_kernel_supports_all_model_shapes():
    """
    Every GroupNorm shape the real U-Net uses must match the torch reference. (Split Kernel)
    """
    # discover real shapes
    counts = survey_groupnorm_shapes(initialize_pipeline())
    torch.manual_seed(0)
    
    for (C, H, W, G) in counts:
        x      = torch.randn(1, C, H, W, dtype=torch.float16, device="cuda")
        weight = torch.randn(C, dtype=torch.float16, device="cuda")
        bias   = torch.randn(C, dtype=torch.float16, device="cuda")
        out = launch_split_config_kernel(x, weight, bias, G, EPS)
        ref = gn_silu_reference(x, weight, bias, G, EPS)
        torch.testing.assert_close(
            out, ref, atol=ATOL, rtol=RTOL,
            msg=lambda m: f"shape (C={C}, H={H}, W={W}, G={G}) mismatch:\n{m}",
        )
    return
