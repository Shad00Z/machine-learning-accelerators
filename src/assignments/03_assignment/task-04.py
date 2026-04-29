import cuda.tile as ct
import random
import torch

from utils import heatmap, next_power_of_two, tileShapeSweep

ConstInt = ct.Constant[int]

# ===========================================================================
# Task 4: L2 Cache Optimization via Block Swizzling
# ===========================================================================

def swizzle_2d_from_bid(M, N, tM, tN, GROUP_SIZE, bid):
    num_tiles_m = ct.cdiv(M, tM)
    num_tiles_n = ct.cdiv(N, tN)

    # Matrix A
    # Row-Group: number of tiles in a group
    num_row_group_tiles = GROUP_SIZE * num_tiles_n
    
    # Row-Group + "general" ID in a Row-Group
    row_group_id     = bid // num_row_group_tiles
    bid_in_row_group = bid %  num_row_group_tiles

    # Start of current Row-Group
    first_bid_m = row_group_id * GROUP_SIZE
    
    # Clamping
    clamp_m = min(num_tiles_m - first_bid_m, GROUP_SIZE)
    
    # Block-Group: 
    mn_block_group = clamp_m * GROUP_SIZE
    
    # Column-Group + "general" ID in a Column-Group
    col_group_id    = bid_in_row_group // mn_block_group
    col_id_in_group = bid_in_row_group %  mn_block_group

    # Start of the current Column-Group
    first_bid_n = col_group_id * GROUP_SIZE
    
    # Clamping
    clamp_n = min(num_tiles_n - first_bid_n, GROUP_SIZE)

    # New block IDs for m and n
    bid_m = first_bid_m + col_id_in_group // clamp_n
    bid_n = first_bid_n + col_id_in_group %  clamp_n

    return bid_m, bid_n


@ct.kernel
def matmul_swizzle_kernel(A, B, C,
                          M: ConstInt, N: ConstInt,
                          tM: ConstInt, tN: ConstInt, tK: ConstInt,
                          GROUP_SIZE_M: ConstInt):
    bid = ct.bid(0)
    
    num_tiles_k = ct.cdiv(A.shape[1], tK)
    bid_m, bid_n = swizzle_2d_from_bid(M, N, tM, tN, GROUP_SIZE_M, bid)

    acc = ct.zeros((tM, tN), dtype=ct.float32)

    for k in range(num_tiles_k):
        a = ct.load(A, index=(bid_m, k), shape=(tM, tK), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, bid_n), shape=(tK, tN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)

    ct.store(C, index=(bid_m, bid_n), tile=ct.astype(acc, C.dtype))


def main():
    # Initialize random tensors
    m = random.randint(1, 4097)
    n = random.randint(1, 4097)
    k = 4096
    
    A = torch.rand(m, k, dtype=torch.float16, device="cuda")
    B = torch.rand(k, n, dtype=torch.float16, device="cuda")
    C = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    
    # Initialize tiles sizes
    tM = next_power_of_two(random.randint(1, 65))
    tN = next_power_of_two(random.randint(1, 65))
    tK = next_power_of_two(random.randint(1, 65))
    
    print(f"Matrix sizes: m={m}, n={n}, k={k}")
    print(f"Tile sizes: tM={tM}, tN={tN}, tK={tK}")
    
    # Kernel Setup
    grid = (ct.cdiv(n, tN) * ct.cdiv(m, tM), 1, 1)
    
    ct.launch(torch.cuda.current_stream(),
              grid,
              matmul_swizzle_kernel,
              (A, B, C, m, n, tM, tN, tK, 8))
    C_torch = torch.matmul(A, B)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Simple Matmul failed!"
    
    # Benchmarking
    # Collecting results for tile shapes(27)
    tile_shapes = [(s, s, s) for s in [256, 512, 2048]]
    
    for m, n, k in tile_shapes:
        tileShapeSweep(m, n, k, matmul_swizzle_kernel, f"task4_{m}_tile_shapes.csv", 8)
        
    # Benchmarking
    for g_size in [4, 8]:
        tileShapeSweep(8192, 8192, 4096, matmul_swizzle_kernel, f"task4_8192_tile_shapes_group_{g_size}.csv", g_size)


if __name__ == "__main__":
    main()
