import pytest
import torch

from sd_turbo.resnet.reference import gn_silu_reference
from sd_turbo.resnet.kernels.gn_silu import launch_reference_config_kernel
from data.load_helper import data_path, load_data
from utils.helper import _cutile_available

ATOL, RTOL = 2e-2, 2e-2


@pytest.fixture(scope="module")
def gn_silu_data():
    """
    Defines a reliable and consistent context for the tests.
    """
    path = data_path()
    if not path.exists():
        pytest.skip(f"reference not found: {path}")
    return load_data(path)


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available")
def test_gn_silu_reference(gn_silu_data):
    """
    Recompute the captured chain.
    """
    y_ref = gn_silu_reference(
        gn_silu_data["x"],
        gn_silu_data["weight"],
        gn_silu_data["bias"],
        gn_silu_data["num_groups"],
        gn_silu_data["eps"],
    )
    assert y_ref.shape == gn_silu_data["y"].shape, "Reference kernel failed!"
    print("Reference kernel passed!")
    torch.testing.assert_close(y_ref, gn_silu_data["y"], atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not _cutile_available(), reason="cuda.tile / CUDA not available")
def test_whole_gn_silu_kernel(gn_silu_data):
    """
    The cuTile kernel matches the reference on real activations.
    """
    x      = gn_silu_data["x"].cuda()
    weight = gn_silu_data["weight"].cuda()
    bias   = gn_silu_data["bias"].cuda()

    out = launch_reference_config_kernel(
        x,
        weight,
        bias,
        gn_silu_data["num_groups"],
        gn_silu_data["eps"],
    )

    assert out.shape == gn_silu_data["y"].shape
    torch.testing.assert_close(out.cpu(), gn_silu_data["y"], atol=ATOL, rtol=RTOL)
    return
