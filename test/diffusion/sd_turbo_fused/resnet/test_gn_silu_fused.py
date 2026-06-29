import torch

from sd_turbo.resnet.resnet_block import gn_silu_reference
from sd_turbo_fused.resnet.gn_silu_kernel import launch_reference_config_kernel
from data.load_helper import data_path, load_data

ATOL, RTOL = 2e-2, 2e-2

def test_gn_silu_reference(ref):
    """The torch reference recomputes the captured chain (sanity on loader + math)."""
    y_ref = gn_silu_reference(
        ref["x"], ref["weight"], ref["bias"],
        ref["num_groups"], ref["eps"],
    )
    assert y_ref.shape == ref["y"].shape, "Reference kernel failed!"
    print("Reference kernel passed!")
    torch.testing.assert_close(y_ref, ref["y"], atol=ATOL, rtol=RTOL)

    return y_ref


def test_gn_silu_kernel(ref):
    """The cuTile kernel matches the ref on real activations -- the M2 gate."""
    # Reference computation
    y_ref = gn_silu_reference(ref["x"], ref["weight"], ref["bias"], ref["num_groups"], ref["eps"])

    # Move data to GPU
    ref.cuda()

    # Compute cuTile reference
    out = launch_reference_config_kernel(ref["x"], ref["weight"], ref["bias"], ref["num_groups"], ref["eps"])

    assert y_ref.shape == out.shape, "Reference kernel failed!"
    torch.testing.assert_close(out.cpu(), y_ref, atol=ATOL, rtol=RTOL)
