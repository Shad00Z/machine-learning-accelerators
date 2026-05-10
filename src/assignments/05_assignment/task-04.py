import cuda.tile as ct
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import torch
import triton

from config import Config, ExecType, DimType
from typing import List, Tuple
from utils import next_power_of_two

ConstInt = ct.Constant[int]

# ---------------------------------------------------------------------------
# Task 4 - L2-Optimized Batched Contraction
# ---------------------------------------------------------------------------

@ct.kernel
def reference_kernel(A, B, C, tM: ConstInt, tN: ConstInt, tK: ConstInt):
    bid = ct.bid(0)
    
    n = bid % ct.cdiv(C.shape[2], tN)
    bid = bid // ct.cdiv(C.shape[2], tN)
    
    m = bid % ct.cdiv(C.shape[1], tM)
    bid = bid // ct.cdiv(C.shape[1], tM)
    
    c = bid % C.shape[0]
    
    acc = ct.zeros((tM, tN), dtype=torch.float32)
    
    for k in range(ct.cdiv(A.shape[2], tK)):
        tile_A = ct.load(A, index=(c, m, k), shape=(1, tM, tK), padding_mode=ct.PaddingMode.ZERO)
        tile_B = ct.load(B, index=(c, k, n), shape=(1, tK, tN), padding_mode=ct.PaddingMode.ZERO)
        
        # Reshape due to rank mismatch
        r_tile_A = ct.reshape(tile_A, (tM, tK))
        r_tile_B = ct.reshape(tile_B, (tK, tN))
        
        acc = ct.mma(r_tile_A, r_tile_B, acc)
            
    o_acc = ct.reshape(acc.astype(C.dtype), (1, tM, tN))
    ct.store(C, index=(c, m, n), tile=o_acc)
    return


def launch_reference_config_kernel(A, B, C, tM, tN, tK):     
    grid = (C.shape[0] * ct.cdiv(C.shape[1], tM) * ct.cdiv(C.shape[2], tN), 1, 1)

    ct.launch(torch.cuda.current_stream(),
              grid,
              reference_kernel,
              (A, B, C, tM, tN, tK))
    return


@ct.kernel
def optimized_config_kernel(A, B, C, tM: ConstInt, tN: ConstInt, tK: ConstInt):
    bid = ct.bid(0)
    
    n1 = bid % ct.cdiv(C.shape[4], tN)
    bid = bid // ct.cdiv(C.shape[4], tN)
    
    m1 = bid % ct.cdiv(C.shape[2], tM)
    bid = bid // ct.cdiv(C.shape[2], tM)
    
    n2 = bid % C.shape[3]
    bid = bid // C.shape[3]
    
    m2 = bid % C.shape[1]
    bid = bid // C.shape[1]
    
    c = bid % C.shape[0]
    
    acc = ct.zeros((tM, tN), dtype=torch.float32)
    
    for k in range(ct.cdiv(A.shape[3], tK)):
        tile_A = ct.load(A, index=(c, m2, m1, k), shape=(1, 1, tM, tK), padding_mode=ct.PaddingMode.ZERO)
        tile_B = ct.load(B, index=(c, k, n2, n1), shape=(1, tK, 1, tN), padding_mode=ct.PaddingMode.ZERO)
        
        # Reshape due to rank mismatch
        r_tile_A = ct.reshape(tile_A, (tM, tK))
        r_tile_B = ct.reshape(tile_B, (tK, tN))
        
        acc = ct.mma(r_tile_A, r_tile_B, acc)
            
    o_acc = ct.reshape(acc.astype(C.dtype), (1, 1, tM, 1, tN))
    ct.store(C, index=(c, m2, m1, n2, n1), tile=o_acc)
    return


def launch_optimized_config_kernel(A, B, C, tM, tN, tK, cfg: Config):
    
    d1_grid = 1
    
    # Step 2: Calculate the grid from cfg
    for dim, dim_type, exec_type, stride in zip(cfg.dim_sizes, cfg.dim_types, cfg.exec_types, cfg.strides[2]):
        if stride == 0:
            continue
        
        if exec_type == ExecType.PRIM:
            if dim_type == DimType.M:
                d1_grid *= ct.cdiv(dim, tM)
            elif dim_type == DimType.N:
                d1_grid *= ct.cdiv(dim, tN)
        else:
            d1_grid *= dim
         
    grid = (d1_grid, 1, 1)
    
    # Step 3: infer new shapes for tensors from cfg
    res_A = res_B = res_C = None
    
    def _infer_tensor_shapes(strides: List[int], dim_sizes: List[int]) -> Tuple[int, ...]:
        """Infers the tensor shapes from the strides of the config object.

        - strides (List[int]): strides for a tensor from the config object.
        - dim_sizes (List[int]): dim sizes from the config object.

        Returns:
            Tuple[int, ...]: dim sizes at the new positions.
        """
        # Step 1: Only keep strides that appear in tensor 
        non_zero = [(idx, stride) for idx, stride in enumerate(strides) if stride != 0]
        
        # Step 2: sort descending by stride value
        sorted_dims = sorted(non_zero, key=lambda x: x[1], reverse=True)
        
        # Step 3: Create list with dim_sizes in new order
        shape = tuple(dim_sizes[idx] for idx, _ in sorted_dims)
        return shape
    
    # Step 5: Reshape the tensors according to the config object
    C_original_shape = C.shape
    for i, stride_list in enumerate(cfg.strides):
        shape = _infer_tensor_shapes(stride_list, cfg.dim_sizes)
        
        if i == 0:
            res_A = torch.reshape(A, shape)
        elif i == 1:
            res_B = torch.reshape(B, shape)
        else:
            res_C = torch.reshape(C, shape)

    # Step 5: Launch the kernel
    ct.launch(torch.cuda.current_stream(),
              grid,
              optimized_config_kernel,
              (res_A, res_B, res_C, tM, tN, tK))
    
    C.copy_(res_C.reshape(C_original_shape))
    return


@ct.kernel
def optimal_config_kernel(A, B, C,
                tM: ConstInt, tN: ConstInt, tK: ConstInt,
                m_in_size: ConstInt, n_in_size: ConstInt):
    """2D super-tile kernel on original 3D tensors — no reshape required.

    BID ordering (innermost → outermost): n_in, m_in, n_out, m_out, c.
    For a fixed (c, m_out, n_out) super-tile all m_in × n_in blocks are
    consecutive, so B[c, :, n_out-group] stays hot in L2 across all m_in tiles
    and A[c, m_out-group, :] stays hot across all n_in tiles.
    """
    bid = ct.bid(0)

    n1 = bid % ct.cdiv(n_in_size, tN)
    bid = bid // ct.cdiv(n_in_size, tN)   # tile within n_in group
    
    m1 = bid % ct.cdiv(m_in_size, tM)
    bid = bid // ct.cdiv(m_in_size, tM)   # tile within m_in group
    
    n2 = bid % ct.cdiv(C.shape[2], n_in_size)
    bid = bid // ct.cdiv(C.shape[2], n_in_size)  # n L2 group
    
    m2 = bid % ct.cdiv(C.shape[1], m_in_size)
    bid = bid // ct.cdiv(C.shape[1], m_in_size)  # m L2 group
    
    c  = bid

    # Flat tile indices — one multiply + add per dimension, ConstInt-folded
    m_tile = m2 * ct.cdiv(m_in_size, tM) + m1
    n_tile = n2 * ct.cdiv(n_in_size, tN) + n1

    acc = ct.zeros((tM, tN), dtype=torch.float32)

    for k in range(ct.cdiv(A.shape[2], tK)):
        tile_A = ct.load(A, index=(c, m_tile, k), shape=(1, tM, tK), padding_mode=ct.PaddingMode.ZERO)
        tile_B = ct.load(B, index=(c, k, n_tile), shape=(1, tK, tN), padding_mode=ct.PaddingMode.ZERO)
        
        acc = ct.mma(ct.reshape(tile_A, (tM, tK)),
                     ct.reshape(tile_B, (tK, tN)),
                     acc)

    ct.store(C, index=(c, m_tile, n_tile), tile=ct.reshape(acc.astype(C.dtype), (1, tM, tN)))
    return


def launch_optimal_config_kernel(A, B, C, tM, tN, tK, m_prim_size=1024, n_prim_size=1024):
    m_prim_tiles = ct.cdiv(m_prim_size, tM)
    n_prim_tiles = ct.cdiv(n_prim_size, tN)
    m_par_count  = ct.cdiv(C.shape[1], m_prim_size)
    n_par_count  = ct.cdiv(C.shape[2], n_prim_size)

    grid = (C.shape[0] * m_par_count * n_par_count * m_prim_tiles * n_prim_tiles, 1, 1)

    ct.launch(torch.cuda.current_stream(), grid, optimal_config_kernel,
              (A, B, C, tM, tN, tK, m_prim_size, n_prim_size))
    return


def benchmarking(launch_fn, out_file, c, m, n, k):
    tile_sizes = list(itertools.product([16, 32, 64, 128], repeat=3))

    records = []
    warmup = 200
    rep    = 2000

    for tM, tN, tK in tile_sizes:
        # bind loop variables explicitly so the lambda captures values, not names
        time_ms = triton.testing.do_bench(
            lambda tM=tM, tN=tN, tK=tK: launch_fn(tM, tN, tK),
            warmup=warmup,
            rep=rep,
        )

        tflops = (2 * c * m * n * k / 1e12) / (time_ms / 1e3)
        print(f"c={c}, m=n=k={m}, tm={tM}, tn={tN}, tk={tK}: {tflops:.2f} TFLOPS")

        records.append({"dim_size": m, "tM": tM, "tN": tN, "tK": tK, "tflops": tflops})

    pd.DataFrame(records).to_csv(out_file, index=False)
    

def heatmap(tK, in_file, out_file):
    df = pd.read_csv(in_file)
    
    heatmap_data = (
        df[df["tK"] == tK]
        .pivot(index="tM", columns="tN", values="tflops")
    )
    values = heatmap_data.values
    
    _, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap_data.values, cmap="plasma")
    
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticklabels(heatmap_data.index)
    ax.set_xlabel("n_tile")
    ax.set_ylabel("m_tile")
    ax.set_title(f"Tile Shape Throughput (TFLOPS)\nc=4, m=n=k=4096, k_tile={tK} fixed")
    
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.1f}",
                    ha="center", va="center",
                    color="white" if values[i, j] < values.max() * 0.7 else "black",
                    fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="TFLOPS")
    plt.tight_layout()

    file_name = out_file
    plt.savefig(file_name, dpi=160)
    print(f"Saved heatmap to {file_name}")
    

if __name__ == "__main__":
    from config import generate_config
    from optimizer import Optimizer
    
    c = 4
    m, n, k = 4096, 4096, 4096
    
    # a)
    cfg = generate_config("cmk,ckn->cmn", [(c, m, k), (c, k, n)])
    print("Task a:")
    print(cfg)
    
    # b)
    #    C    M    K    N
    #    4 4096 4096 4096
    opt = Optimizer(cfg)
    
    # L2 Size: 24MiB = 24 * 1024**2
    l2_size = 24 * 1024**2
    print(f"L2 Cache Size: {l2_size}\n")
    
    # (M*K + K*N) * 2 + (MN) * 4: (4096 * 4096 + 4096 * 4096) * 2 + (4096 * 4096) * 4
    cur_cache_use = (m * k + k * n) * 2 + (m * n) * 4
    print(f"Original memory load: {cur_cache_use}")
    print(f"Comparing to 24MiB: {cur_cache_use - l2_size}\n")
    
    # Split N dimension
    n1, n2 = 4, 1024
    opt.split_dim(3, n1, n2)
    cur_cache_use = (m * k + k * n2) * 2 + (m * n2) * 4
    print(f"Memory load after splitting M dimension: {cur_cache_use}")
    print(f"Comparing to 24MiB: l2_size - memory load = {l2_size - cur_cache_use - l2_size} free\n")
    
    # Split M dimension
    m1, m2 = 4, 1024
    opt.split_dim(1, m1, m2)
    cur_cache_use = (m2 * k + k * n2) * 2 + (m2 * n2) * 4
    print(f"Memory load after splitting M dimension: {cur_cache_use}")
    print(f"Comparing to 24MiB: l2_size - memory load = {l2_size - cur_cache_use} free\n")
    
    opt.make_executable()
    print(cfg)
    
    # c) launch kernel
    A = torch.rand((c, m, k), dtype=torch.float16, device="cuda")
    B = torch.rand((c, k, n), dtype=torch.float16, device="cuda")
    C = torch.zeros((c, m, n), dtype=torch.float32, device="cuda")
    
    tM = tN = tK = 0
    
    # Step 1: Calculate tile sizes
    for dim_size, dim_type, exec_type in zip(cfg.dim_sizes, cfg.dim_types, cfg.exec_types):
        if exec_type == ExecType.PRIM:
            tile = next_power_of_two(dim_size // 16)
            
            if dim_type == DimType.M:
                tM = tile
            elif dim_type == DimType.N:
                tN = tile
            else:
                tK = tile
    launch_optimized_config_kernel(A, B, C, tM, tN, tK, cfg)
    
    C_torch = torch.einsum('cmk,ckn->cmn', A.float(), B.float())

    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Task 4c failed!"
    print("Kernel 4c passed!")
    
    # d) benchmarking
    C = torch.zeros_like(C)
    launch_optimal_config_kernel(A, B, C, tM, tN, tK)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Lean kernel failed!"
    print("Lean kernel passed!")
    
    C = torch.zeros_like(C)
    launch_reference_config_kernel(A, B, C, tM, tN, tK)
    
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Task 4c reference failed!"
    print("Kernel 4c reference passed!")
    
    path = "src/assignments/05_assignment/resources-05"
    benchmarking(
        lambda tM, tN, tK: launch_optimized_config_kernel(A, B, C, tM, tN, tK, cfg),
        f"{path}/task4_optimizer.csv", c, m, n, k
    )
    benchmarking(
        lambda tM, tN, tK: launch_reference_config_kernel(A, B, C, tM, tN, tK),
        f"{path}/task4_reference.csv", c, m, n, k
    )
    benchmarking(
        lambda tM, tN, tK: launch_optimal_config_kernel(A, B, C, tM, tN, tK),
        f"{path}/task4_lean.csv", c, m, n, k
    )
    for tK in [16, 32, 64, 128]:
        heatmap(tK, f"{path}/task4_optimizer.csv", 
                f"{path}/task4_k={tK}_heatmap.png")
        heatmap(tK, f"{path}/task4_reference.csv", 
                f"{path}/task4_k={tK}_heatmap_ref.png")
        heatmap(tK, f"{path}/task4_lean.csv",
                f"{path}/task4_k={tK}_heatmap_lean.png")
