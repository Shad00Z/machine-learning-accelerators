import numpy as np
import pytest
import torch

from sd_turbo.transformer.transformer_block import ffn_reference
from data.load_helper import data_path
from utils.helper import _cutile_available

# fp16 through two matmuls: measured max_abs ~3e-4
ATOL, RTOL = 3e-2, 3e-2


##################################################
# Setup Tests
##################################################
def ffn_ref_path():
    return data_path().parent / "ffn_block0.npz"


def load_ffn_reference(path):
    z = np.load(path)
    d = {k: torch.from_numpy(z[k]) for k in ("x", "y", "ln_weight", "ln_bias", "w1", "b1", "w2", "b2")}
    d["eps"] = float(z["eps"])
    d["dim"] = int(z["dim"])
    return d


@pytest.fixture(scope="module")
def ffn_data():
    p = ffn_ref_path()
    if not p.exists():
        pytest.skip(f"FFN reference not found: {p} -- run test/diffusion/data/ffn_generation.py")
    return load_ffn_reference(p)


def test_ffn_reference_matches_captured(ffn_data):
    """Torch reference reproduces the captured chain output (loader + math sanity)."""
    f = lambda k: ffn_data[k].float()
    y = ffn_reference(f("x"), f("ln_weight"), f("ln_bias"), ffn_data["eps"],
                      f("w1"), f("b1"), f("w2"), f("b2"))
    
    assert y.shape == ffn_data["y"].shape
    torch.testing.assert_close(y.to(ffn_data["y"].dtype), ffn_data["y"], atol=ATOL, rtol=RTOL)
    return


##################################################
# Launch Kernel Tests
##################################################

@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available (run on the Spark)")
def test_ffn_kernel_matches_reference(ffn_data):
    """The fused cuTile FFN kernel matches the captured output."""
    from sd_turbo_fused.transformer.ffn_kernel import launch_ffn_kernel

    cu = lambda k: ffn_data[k].cuda()
    out = launch_ffn_kernel(cu("x"), cu("ln_weight"), cu("ln_bias"), ffn_data["eps"],
                            cu("w1"), cu("b1"), cu("w2"), cu("b2"))
    
    torch.testing.assert_close(out.cpu(), ffn_data["y"], atol=ATOL, rtol=RTOL)
    return


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available (run on the Spark)")
def test_ffn_gemm_split_matches_reference(ffn_data):
    """The (LN,mm1)+(GEGLU,mm2) split matches the captured output."""
    from sd_turbo_fused.transformer.ffn_gemm_split_kernel import launch_ffn_gemm_split

    cu = lambda k: ffn_data[k].cuda()
    out = launch_ffn_gemm_split(cu("x"), cu("ln_weight"), cu("ln_bias"), ffn_data["eps"],
                                cu("w1"), cu("b1"), cu("w2"), cu("b2"))
    
    torch.testing.assert_close(out.cpu(), ffn_data["y"], atol=ATOL, rtol=RTOL)
    return
    
    
@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available (run on the Spark)")
def test_ffn_swizzle_matches_reference(ffn_data):
    """The L2-swizzled FFN matches the captured output."""
    from sd_turbo_fused.transformer.ffn_swizzle_kernel import launch_ffn_swizzle

    cu = lambda k: ffn_data[k].cuda()
    out = launch_ffn_swizzle(cu("x"), cu("ln_weight"), cu("ln_bias"), ffn_data["eps"],
                             cu("w1"), cu("b1"), cu("w2"), cu("b2"))
    
    torch.testing.assert_close(out.cpu(), ffn_data["y"], atol=ATOL, rtol=RTOL)
    return


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available (run on the Spark)")
def test_ffn_split_matches_reference(ffn_data):
    """The single-accumulator split FFN matches the captured output."""
    from sd_turbo_fused.transformer.ffn_split_kernel import launch_ffn_split

    cu = lambda k: ffn_data[k].cuda()
    out = launch_ffn_split(cu("x"), cu("ln_weight"), cu("ln_bias"), ffn_data["eps"],
                           cu("w1"), cu("b1"), cu("w2"), cu("b2"))
    
    torch.testing.assert_close(out.cpu(), ffn_data["y"], atol=ATOL, rtol=RTOL)
    return


##################################################
# Single Kernel Tests
##################################################

@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available (run on the Spark)")
def test_mm1_ln_matches_reference(ffn_data):
    """Stage 2: mm1 + LayerNorm first touch matches F.linear(F.layer_norm(x)) on a real block."""
    import torch.nn.functional as F
    from sd_turbo_fused.transformer.ffn_gemm_split_kernel import launch_mm1_ln

    dim = ffn_data["dim"]
    cu = lambda k: ffn_data[k].cuda()
    x2d = cu("x").reshape(-1, dim)
    proj = launch_mm1_ln(x2d, cu("ln_weight"), cu("ln_bias"), ffn_data["eps"], cu("w1"), cu("b1"))
    ref = F.linear(
        F.layer_norm(x2d.float(), (dim,), cu("ln_weight").float(), cu("ln_bias").float(), ffn_data["eps"]),
        cu("w1").float(), cu("b1").float(),
    )
    
    torch.testing.assert_close(proj.float().cpu(), ref.float().cpu(), atol=1e-1, rtol=2e-2)
    return


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available (run on the Spark)")
def test_mm2_matches_linear():
    """The ct.mma mm2 kernel matches F.linear on real FFN dims."""
    from sd_turbo_fused.transformer.ffn_kernel import launch_mm2

    torch.manual_seed(0)
    M, inner, dim = 4096, 1280, 320
    gated = torch.randn(M, inner, dtype=torch.float16, device="cuda")
    w2    = torch.randn(dim, inner, dtype=torch.float16, device="cuda")
    b2    = torch.randn(dim,        dtype=torch.float16, device="cuda")

    out = launch_mm2(gated, w2, b2)
    ref = torch.nn.functional.linear(gated, w2, b2)
    
    torch.testing.assert_close(out.float().cpu(), ref.float().cpu(), atol=1e-1, rtol=2e-2)
    return


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available (run on the Spark)")
def test_mm1_geglu_matches_reference(ffn_data):
    """Kernel A (LN -> mm1 -> GEGLU) matches the reference on a real block."""
    import torch.nn.functional as F
    from sd_turbo_fused.transformer.ffn_kernel import launch_mm1_geglu

    dim = ffn_data["dim"]
    cu = lambda k: ffn_data[k].cuda()
    x2d = cu("x").reshape(-1, dim)
    gated = launch_mm1_geglu(x2d, cu("ln_weight"), cu("ln_bias"), ffn_data["eps"], cu("w1"), cu("b1"))

    normed = F.layer_norm(x2d.float(), (dim,), cu("ln_weight").float(), cu("ln_bias").float(), ffn_data["eps"])
    proj = F.linear(normed, cu("w1").float(), cu("b1").float())
    hidden, gate = proj.chunk(2, dim=-1)
    ref = hidden * F.gelu(gate)
    torch.testing.assert_close(gated.float().cpu(), ref.float().cpu(), atol=1e-1, rtol=2e-2)
    return
