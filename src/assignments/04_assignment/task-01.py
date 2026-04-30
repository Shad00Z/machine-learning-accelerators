import cuda.tile as ct
import random
import torch
import triton

from utils import next_power_of_two, tensor_initialization

ConstInt = ct.Constant[int]

# ===========================================================================
# Task 1b: Sequentialize k and l
# ===========================================================================


@ct.kernel
def tile_contraction_kl(A, B, C, tM: ConstInt, tN: ConstInt, tK: ConstInt):
    bid = ct.bid(0)
    
    # Indices C: Right to left
    z = bid % ct.cdiv(C.shape[5], tN)
    bid = bid // ct.cdiv(C.shape[5], tN)
    
    x = bid % ct.cdiv(C.shape[4], tM)
    bid = bid // ct.cdiv(C.shape[4], tM)
    
    c = bid % C.shape[3]
    bid = bid // C.shape[3]
    
    b = bid % C.shape[2]
    bid = bid // C.shape[2]
    
    a = bid % C.shape[1]
    bid = bid // C.shape[1]
    
    e = bid % C.shape[0]
    
    acc = ct.zeros((tM, tN), dtype=torch.float32)
    
    for k in range(A.shape[3]):
        for l in range(A.shape[4]):
            for y in range(A.shape[6]):
                tile_A = ct.load(A, index=(e, a, b, k, l, x, y), shape=(1, 1, 1, 1, 1, tM, tK), padding_mode=ct.PaddingMode.ZERO)
                tile_B = ct.load(B, index=(e, c, k, l, y, z),    shape=(1, 1, 1, 1, tK, tN),    padding_mode=ct.PaddingMode.ZERO)
                
                # Reshape due to rank mismatch
                r_tile_A = ct.reshape(tile_A, (tM, tK))
                r_tile_B = ct.reshape(tile_B, (tK, tN))
                
                acc = ct.mma(r_tile_A, r_tile_B, acc)
            
    o_acc = ct.reshape(acc.astype(C.dtype), (1, 1, 1, 1, tM, tN))
    ct.store(C, index=(e, a, b, c, x, z), tile=o_acc)
    return


def main():
    A, B, C = tensor_initialization()
    
    # Initialize tiles sizes
    tM = next_power_of_two(A.shape[5] // 2)
    tN = next_power_of_two(B.shape[5] // 2)
    tK = next_power_of_two(B.shape[4] // 2)
    
    print(f"Matrix shapes: A={A.size()}, B={B.size()}, C={C.size()}")
    print(f"Tile sizes: tM={tM}, tN={tN}, tK={tK}")
    
    # eabcxz
    grid = (C.shape[0] * C.shape[1] * C.shape[2] * C.shape[3] * ct.cdiv(C.shape[4], tM) * ct.cdiv(C.shape[5], tN), 1, 1)

    ct.launch(torch.cuda.current_stream(),
              grid,
              tile_contraction_kl,
              (A, B, C, tM, tN, tK))

    C_torch = torch.einsum('eabklxy, ecklyz -> eabcxz', A, B)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Simple Matmul failed!"
    
    return


if __name__ == "__main__":
    main()
