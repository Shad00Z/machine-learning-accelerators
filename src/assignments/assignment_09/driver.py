"""
XRT Python driver for Assignment 09.

Usage (from the assignment directory, after building xclbins):
    python3 src/driver.py

Requires: pyxrt, numpy, torch
"""

import numpy as np
import torch
import pyxrt


def verify(in0: torch.Tensor, in1: torch.Tensor, out: torch.Tensor) -> None:
    """
    Verify the NPU output against a CPU reference.

    Computation: out += in0 @ in1

    Parameters
    ----------
    in0, in1 : bfloat16 torch tensors
    out : bfloat16 torch tensor
    """
    ref = in0 @ in1

    print("\n" + "="*40)
    print("       NPU DEBUG DIAGNOSTICS        ")
    print("="*40)
    
    # 1. Check for complete silence (All Zeros)
    is_all_zero = torch.all(out == 0).item()
    print(f"Is output completely zeros?: {is_all_zero}")
    
    # 2. Check fill percentage
    nonzero_count = torch.count_nonzero(out).item()
    total_elements = out.numel()
    print(f"Active elements: {nonzero_count} / {total_elements} ({nonzero_count/total_elements*100:.2f}%)")
    
    # 3. Value Range Comparison
    print(f"NPU Output Range: Min = {out.min().item():.4f}, Max = {out.max().item():.4f}")
    print(f"CPU Reference Range: Min = {ref.min().item():.4f}, Max = {ref.max().item():.4f}")
    
    # 4. Error Metrics
    abs_diff = torch.abs(out - ref)
    print(f"Max Absolute Error: {abs_diff.max().item():.4f}")
    print(f"Mean Absolute Error: {abs_diff.mean().item():.4f}")
    
    # 5. Row-by-Row Spatial Analysis
    failing_rows = []
    for r in range(out.shape[0]):
        if not torch.allclose(out[r], ref[r], rtol=0.5, atol=2):
            failing_rows.append(r)
            
    print(f"Failing Rows: {len(failing_rows)} / {out.shape[0]}")
    if len(failing_rows) > 0:
        print(f"First 10 failing row indices: {failing_rows[:10]}")
    print("="*40 + "\n")

    if not torch.allclose(out, ref, rtol=0.5, atol=2):
        raise ValueError(f"[FAIL] verification did not pass.")


def run() -> None:
    xclbin_path = "build/final_matmul.xclbin"
    insts_path = "build/insts_matmul.bin"

    insts = np.fromfile(insts_path, dtype=np.uint32)

    device = pyxrt.device(0)
    xclbin = pyxrt.xclbin(xclbin_path)
    device.register_xclbin(xclbin)
    uuid = xclbin.get_uuid()
    context = pyxrt.hw_context(device, uuid)
    kname = xclbin.get_kernels()[0].get_name()
    kernel = pyxrt.kernel(context, kname)

    bo_instr = pyxrt.bo(device, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    bo_instr.write(insts.tobytes(), 0)
    bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, insts.nbytes, 0)

    torch.manual_seed(42)
    # M=256, N=128, K=1024
    data_in0 = torch.randn( 256, 1024, dtype=torch.bfloat16)
    data_in1 = torch.randn(1024,  128, dtype=torch.bfloat16)
    data_out = torch.zeros( 256,  128, dtype=torch.bfloat16)

    # Create buffer objects with corresponding size
    bo_in0 = pyxrt.bo(device, data_in0.nbytes, pyxrt.bo.host_only, 0)
    bo_in1 = pyxrt.bo(device, data_in1.nbytes, pyxrt.bo.host_only, 0)
    bo_out = pyxrt.bo(device, data_out.nbytes, pyxrt.bo.host_only, 0)

    # Copy data to buffer objects
    bo_in0.write(data_in0.view(torch.int16).numpy().tobytes(), 0)
    bo_in1.write(data_in1.view(torch.int16).numpy().tobytes(), 0)
    bo_out.write(data_out.view(torch.int16).numpy().tobytes(), 0)

    # View buffer objects as torch tensor
    tensor_in0 = torch.frombuffer(
        bo_in0.map(),
        dtype=torch.bfloat16,
        count=np.prod(data_in0.shape)
    ).view(data_in0.shape)
    tensor_in1 = torch.frombuffer(
        bo_in1.map(),
        dtype=torch.bfloat16,
        count=np.prod(data_in1.shape)
    ).view(data_in1.shape)
    tensor_out = torch.frombuffer(
        bo_out.map(),
        dtype=torch.bfloat16,
        count=np.prod(data_out.shape)
    ).view(data_out.shape)
    assert torch.equal(data_in0, tensor_in0)
    assert torch.equal(data_in1, tensor_in1)
    assert torch.equal(data_out, tensor_out)

    # Sync buffer objects: to device
    bo_in0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in0.nbytes, 0)
    bo_in1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in1.nbytes, 0)
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_out.nbytes, 0)

    h = kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)
    h.wait()

    # Sync output buffer object: from device
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, data_out.nbytes, 0)

    verify(tensor_in0, tensor_in1, tensor_out)

    print("[PASS] matmul verification passed.")


if __name__ == "__main__":
    run()