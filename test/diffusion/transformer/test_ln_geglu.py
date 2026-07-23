import math

import pytest
import torch

from sd_turbo.transformer.reference import ln_geglu_reference
from sd_turbo.transformer.kernels.ln_geglu import launch_ln_geglu_mm1
from utils.helper import _cutile_available

ATOL, RTOL = 1e-2, 1e-2

T, K, N1 = 77, 320, 2560
EPS = 1e-5


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available")
def test_ln_geglu_reference():
    torch.manual_seed(0)
    x      = torch.randn(T, K,  dtype=torch.float16, device="cpu")
    ln_w   = torch.randn(K,     dtype=torch.float16, device="cpu") * 0.1 + 1.0
    ln_b   = torch.randn(K,     dtype=torch.float16, device="cpu") * 0.1
    W1     = torch.randn(N1, K, dtype=torch.float16, device="cpu") / math.sqrt(K)
    b1     = torch.randn(N1,    dtype=torch.float16, device="cpu") * 0.1

    y_ref = ln_geglu_reference(x, ln_w, ln_b, W1, b1, EPS)
    assert y_ref.shape == (T, N1 // 2), "Reference kernel failed!"
    print("Reference kernel passed!")


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available")
def test_ln_geglu_kernel():
    torch.manual_seed(0)
    x      = torch.randn(T, K,  dtype=torch.float16, device="cpu")
    ln_w   = torch.randn(K,     dtype=torch.float16, device="cpu") * 0.1 + 1.0
    ln_b   = torch.randn(K,     dtype=torch.float16, device="cpu") * 0.1
    W1     = torch.randn(N1, K, dtype=torch.float16, device="cpu") / math.sqrt(K)
    b1     = torch.randn(N1,    dtype=torch.float16, device="cpu") * 0.1

    y_ref = ln_geglu_reference(x, ln_w, ln_b, W1, b1, EPS)

    x, ln_w, ln_b, W1, b1 = x.cuda(), ln_w.cuda(), ln_b.cuda(), W1.cuda(), b1.cuda()
    out = launch_ln_geglu_mm1(x, ln_w, ln_b, W1, b1, EPS)

    assert y_ref.shape == out.shape, "Kernel shape mismatch!"
    torch.testing.assert_close(out.cpu(), y_ref, atol=ATOL, rtol=RTOL)
