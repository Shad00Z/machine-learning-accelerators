import cuda.tile as ct
import numpy as np
import torch
import opt_einsum # unused but required for torch.einsum memory optimization
import matplotlib.pyplot as plt
import triton

from src.assignments.assignment_05.config import Config, generate_config
from src.assignments.assignment_05.optimizer import Optimizer

ConstInt = ct.Constant[int]

def plot_tensor(
    tensor,
    path='tensor_plot.png',
    title=''
):
    """
    Plots a 5D tensor by slicing along the first two dimensions and displaying the resulting images.
    Dimension order is assumed to be (a, b, c, y, x) where a and b are image indices and c is the color channel.

    Args:
        tensor (torch.Tensor): A 5D tensor of shape (a, b, c, y, x).
        title (str): Title for the plot.
    """
    a, b, c, y, x = tensor.shape
    fig, axes = plt.subplots(a, b, figsize=(b * 2, a * 2))
    for i in range(a):
        for j in range(b):
            img = tensor[i, j].numpy()
            # reorder from c,y,x to y,x,c
            img = np.transpose(img, (1, 2, 0))
            img *= 255.0
            img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    
# -----------------------------------------------------------------------
# Task 4a - cuTile Kernel
# -----------------------------------------------------------------------

@ct.kernel
def multi_input_kernel(A, B, C, tM: ConstInt, tN: ConstInt, tK: ConstInt):
    
    bid = ct.bid(0)
    
    m2_par = bid % C.shape[6]
    bid = bid // C.shape[6]
    
    n_par = bid % C.shape[1]
    bid = bid // C.shape[1]
    
    m1_par = bid % C.shape[0]

    for m1_seq in range(C.shape[5]):
        for m2_seq in range(C.shape[2]):
            for n_seq in range(C.shape[3]):
                acc = ct.zeros((tM, tN), dtype=torch.float32)
                
                for k_seq in range(A.shape[2]):
                    tile_A = ct.load(A, index=(m1_par, m2_seq, k_seq, 0, m1_seq, m2_par, 0), shape=(1, 1, 1, tK, 1, 1, tM), padding_mode=ct.PaddingMode.ZERO)
                    tile_B = ct.load(B, index=(n_par, k_seq, 0, n_seq, 0), shape=(1, 1, tK, 1, tN), padding_mode=ct.PaddingMode.ZERO)
                    
                    # Reshape due to rank mismatch
                    r_tile_A = ct.reshape(tile_A, (tK, tM))
                    r_tile_A = ct.transpose(r_tile_A)
                    r_tile_B = ct.reshape(tile_B, (tK, tN))
                    
                    acc = ct.mma(r_tile_A, r_tile_B, acc)
            
                o_acc = ct.reshape(ct.transpose(acc).astype(C.dtype), (1, 1, 1, 1, tN, 1, 1, tM))
                ct.store(C, index=(m1_par, n_par, m2_seq, n_seq, 0, m1_seq, m2_par, 0), tile=o_acc)
    return


def launch_multi_input_kernel(A, B, C):

    grid = (C.shape[0] * C.shape[1] * C.shape[6], 1, 1)

    ct.launch(torch.cuda.current_stream(),
              grid,
              multi_input_kernel,
              (A, B, C, C.shape[7], C.shape[4], A.shape[3]))
    return
    
# -----------------------------------------------------------------------
# Tasks - Shared Code
# -----------------------------------------------------------------------

def print_kernel_mapping(cfg):
    """Prints exactly which tensor position maps to which role."""
    tensor_names = ["A", "B", "C"]
    for t_idx, name in enumerate(tensor_names):
        t_strides = cfg.strides[t_idx]
        dims = [(i, cfg.dim_sizes[i], cfg.dim_types[i].value,
                 cfg.exec_types[i].value, t_strides[i])
                for i in range(len(cfg.dim_sizes)) if t_strides[i] != 0]
        dims.sort(key=lambda x: -x[4])
        print(f"{name}: shape={[d[1] for d in dims]}")
        for pos, (i, sz, dt, et, st) in enumerate(dims):
            print(f"  [{name}[{pos}]] cfg_dim={i}  size={sz}  type={dt}  exec={et}  stride={st}")
        print()
        

def print_config_information(cfg):
    tensor_names = ["A (input 0)", "B (input 1)", "C (output)"]
    
    for t_idx, (name, t_strides) in enumerate(zip(tensor_names, cfg.strides)):
        # Collect dims that belong to this tensor (non-zero stride)
        dims = [
            (i, cfg.dim_sizes[i], cfg.dim_types[i].value, 
                cfg.exec_types[i].value, t_strides[i])
            for i in range(len(cfg.dim_sizes))
            if t_strides[i] != 0
        ]
        
        # Sort by stride descending = outermost dim first
        dims.sort(key=lambda x: -x[4])
        
        shape = [d[1] for d in dims]
        total = 1
        for s in shape: total *= s
        
        print(f"{name}:")
        print(f"  shape      = {shape}")
        print(f"  total elems= {total:,}")
        print(f"  size bytes = {total * (2 if cfg.data_type.value == 'FLOAT16' else 4):,}")
        print(f"  dims:")
        for (i, size, dtype, etype, stride) in dims:
            print(f"    dim[{i}]: size={size:4d}  type={dtype}  exec={etype:4s}  stride={stride}")
        print()
        

def get_tensor_shape(cfg, t_idx):
    """Return the reshape tuple for tensor t_idx, ordered outermost -> innermost by stride."""
    t_strides = cfg.strides[t_idx]
    
    # Collect dims present in this tensor (non-zero stride)
    dims = [
        (i, cfg.dim_sizes[i], t_strides[i])
        for i in range(len(cfg.dim_sizes))
        if t_strides[i] != 0
    ]
    
    # Sort by stride descending = outermost first
    dims.sort(key=lambda x: -x[2])
    
    return tuple(size for (_, size, _) in dims)


def main():
    # Load last two intermediate tensors from disk
    print("Loading intermediate tensors from disk...")
    path = './src/assignments/assignment_06'
    data = np.load(f'{path}/data/lf_tr_64_intermediate.npz')
    
    # Task 1
    
    # FP32
    tensor_acspx_fp32 = torch.tensor(data['tensor_acspx'], device='cuda:0', dtype=torch.float32)
    tensor_bspy_fp32 = torch.tensor(data['tensor_bspy'], device='cuda:0', dtype=torch.float32)
    tensor_abcyx_fp32 = torch.einsum("acspx,bspy->abcyx", tensor_acspx_fp32, tensor_bspy_fp32).to(device='cpu')
    
    # FP16
    tensor_acspx_fp16 = torch.tensor(data['tensor_acspx'], device='cuda:0', dtype=torch.float16)
    tensor_bspy_fp16 = torch.tensor(data['tensor_bspy'], device='cuda:0', dtype=torch.float16)
    tensor_abcyx_fp16 = torch.einsum("acspx,bspy->abcyx", tensor_acspx_fp16, tensor_bspy_fp16).to(device='cpu')

    # plot_tensor(
    #     tensor_abcyx_fp32,
    #     path=f'{path}/results/torch_32.png',
    #     title='Lightfield Tensorring Decomposition - All Ranks: 64 - PyTorch - FP32'
    # )
    
    # plot_tensor(
    #     tensor_abcyx_fp16,
    #     path=f'{path}/results/torch_16.png',
    #     title='Lightfield Tensorring Decomposition - All Ranks: 64 - PyTorch - FP16'
    # )
    
    # Task 2
    print("Task 2 - Initial Config:")
    cfg = generate_config("acspx,bspy->abcyx", [tensor_acspx_fp16.shape, tensor_bspy_fp16.shape])
    print(cfg)
    
    # Task 3
    #   M   M    K   K     M    N      N
    #   4   3   64  64  1536    4   1152
    print("Task 3 - Optimized Config:")
    opt = Optimizer(cfg)
    
    l2_size = 24 * 1024**2
    print(f"L2 Cache Size: {l2_size}\n")
    
    # Step 1: Find PRIM dimensions
    # M = 128, N = 128, K = 64
    
    n1, n2 = 9, 128
    opt.split_dim(6, n1, n2)
    
    m1, m2 = 12, 128
    opt.split_dim(4, m1, m2)
    opt.make_executable()
    
    # Step 2: L2 Cache Size
    # Split: M = 12 into 3 * 4
    opt.split_dim(2, 3, 4)
    
    # Fuse:  M = 3 and M = 3
    # opt.fuse_dims(1, 2)
    opt.make_executable()
    
    a_tile = (9 * m2) * (64 * 64) * 2
    b_tile = (64 * 64) * (n2 * 9) * 2
    c_tile = (9 * m2) * (n2 * 9) * 4
    
    total_bytes = a_tile + b_tile + c_tile
    diff = l2_size - total_bytes
    assert diff > 0, f"Tile size exceeds L2 cache size by {-diff} bytes"
    print(f"Diff: {diff}")
    
    # Select SEQ dims for m1_l2, m2_l2 and n_l2
    opt.assign_seq_dims([1, 2, 5])
    print(cfg)
    
    print("Kernel mapping")
    print_kernel_mapping(cfg)
    
    # Task 4
    shape_A = get_tensor_shape(cfg, t_idx=0)
    shape_B = get_tensor_shape(cfg, t_idx=1)
    shape_C = get_tensor_shape(cfg, t_idx=2)
    
    print_config_information(cfg)
    
    cutile_tensor_abcyx_fp32 = torch.zeros_like(tensor_abcyx_fp32, device='cuda:0')
    launch_multi_input_kernel(tensor_acspx_fp16.reshape(shape_A), tensor_bspy_fp16.reshape(shape_B), cutile_tensor_abcyx_fp32.reshape(shape_C))
    assert torch.allclose(tensor_abcyx_fp32.to(device='cuda:0').to(torch.float32), cutile_tensor_abcyx_fp32, atol=1e-2), "Task 4b failed"
    print("Kernel 4b passed!")

    # Benchmark - cuTile
    cutile_tensor_abcyx_fp32 = torch.zeros_like(tensor_abcyx_fp32, device='cuda:0')
    warmup, rep = 200, 2000
    cutile_result = triton.testing.do_bench(
        lambda: launch_multi_input_kernel(tensor_acspx_fp16.reshape(shape_A), tensor_bspy_fp16.reshape(shape_B), cutile_tensor_abcyx_fp32.reshape(shape_C)),
        warmup=warmup, rep=rep)
    cutile_tflops = 2 * tensor_acspx_fp16.numel() * tensor_bspy_fp16.shape[2] / (cutile_result / 1000) / 1e12
    
    # Benchmark - PyTorch
    torch_result = triton.testing.do_bench(
        lambda: torch.einsum("acspx,bspy->abcyx", tensor_acspx_fp16, tensor_bspy_fp16),
        warmup=warmup, rep=rep)
    torch_tflops = 2 * tensor_acspx_fp16.numel() * tensor_bspy_fp16.shape[2] / (torch_result / 1000) / 1e12
    
    print(f"cuTile Kernel Average Time: {cutile_result:.2f} ms")
    print(f"PyTorch Average Time: {torch_result:.2f} ms")
    print(f"cuTile Kernel TFLOPS: {cutile_tflops:.2f} TFLOPS")
    print(f"PyTorch TFLOPS: {torch_tflops:.2f} TFLOPS")


if __name__ == "__main__":
    main()
