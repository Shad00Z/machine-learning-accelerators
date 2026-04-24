import cuda.tile as ct
import random
import torch

ConstInt = ct.Constant[int]

# ===========================================================================
# Task 2: Simple Matrix Multiplication Kernel
# ===========================================================================

def next_power_of_two(n: int) -> int:
	p = 1
	while p < n:
		p *= 2
	return p


@ct.kernel
def mma_kernel(A, B, C, tM: ConstInt, tN: ConstInt, tK: ConstInt):
    bid = ct.bid(0)
    
    # index calculation
    num_tiles_k = ct.cdiv(A.shape[1], tK)
    num_tiles_n = ct.cdiv(B.shape[1], tN)
    m = bid // num_tiles_n	# row
    n = bid % num_tiles_n	# col
    
    acc = ct.zeros((tM, tN), dtype=torch.float32)

    for k in range(num_tiles_k):
        tile_A = ct.load(A, index=(m, k), shape=(tM, tK), padding_mode=ct.PaddingMode.ZERO)
        tile_B = ct.load(B, index=(k, n), shape=(tK, tN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(tile_A, tile_B, acc)

    ct.store(C, index=(m, n), tile=acc)
    return


def main():
    # Initialize random tensors
    m = random.randint(1, 4097)
    n = random.randint(1, 4097)
    k = random.randint(1, 4097)
    
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
              mma_kernel,
              (A, B, C, tM, tN, tK))
    C_torch = torch.matmul(A, B)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Simple Matmul failed!"


if __name__ == "__main__":
    main()
