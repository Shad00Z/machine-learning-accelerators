import cuda.tile as ct
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
    
    # Indices C: Right to left (eabcxz)
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
            for y in range(ct.cdiv(A.shape[6], tK)):
                tile_A = ct.load(A, index=(e, a, b, k, l, x, y), shape=(1, 1, 1, 1, 1, tM, tK), padding_mode=ct.PaddingMode.ZERO)
                tile_B = ct.load(B, index=(e, c, k, l, y, z),    shape=(1, 1, 1, 1, tK, tN),    padding_mode=ct.PaddingMode.ZERO)
                
                # Reshape due to rank mismatch
                r_tile_A = ct.reshape(tile_A, (tM, tK))
                r_tile_B = ct.reshape(tile_B, (tK, tN))
                
                acc = ct.mma(r_tile_A, r_tile_B, acc)
            
    o_acc = ct.reshape(acc.astype(C.dtype), (1, 1, 1, 1, tM, tN))
    ct.store(C, index=(e, a, b, c, x, z), tile=o_acc)
    return


# ===========================================================================
# Task 1c: Sequentialize k, l and b
# ===========================================================================

@ct.kernel
def tile_contraction_klb(A, B, C, tM: ConstInt, tN: ConstInt, tK: ConstInt):
    bid = ct.bid(0)

    # Indices C: Right to left (eabcxz)
    z = bid % ct.cdiv(C.shape[5], tN)
    bid = bid // ct.cdiv(C.shape[5], tN)

    x = bid % ct.cdiv(C.shape[4], tM)
    bid = bid // ct.cdiv(C.shape[4], tM)

    c = bid % C.shape[3]
    bid = bid // C.shape[3]

    a = bid % C.shape[1]
    bid = bid // C.shape[1]

    e = bid % C.shape[0]

    for b in range(A.shape[2]):
        acc = ct.zeros((tM, tN), dtype=torch.float32)

        for k in range(A.shape[3]):
            for l in range(A.shape[4]):
                for y in range(ct.cdiv(A.shape[6], tK)):
                    tile_A = ct.load(A, index=(e, a, b, k, l, x, y), shape=(1, 1, 1, 1, 1, tM, tK), padding_mode=ct.PaddingMode.ZERO)
                    tile_B = ct.load(B, index=(e, c, k, l, y, z),    shape=(1, 1, 1, 1, tK, tN),    padding_mode=ct.PaddingMode.ZERO)

                    r_tile_A = ct.reshape(tile_A, (tM, tK))
                    r_tile_B = ct.reshape(tile_B, (tK, tN))

                    acc = ct.mma(r_tile_A, r_tile_B, acc)

        o_acc = ct.reshape(acc.astype(C.dtype), (1, 1, 1, 1, tM, tN))
        ct.store(C, index=(e, a, b, c, x, z), tile=o_acc)

    return


def launch_tile_contraction_kl(A, B, C, tM, tN, tK):
    # eabcxz
    grid = (C.shape[0] * C.shape[1] * C.shape[2] * C.shape[3] * ct.cdiv(C.shape[4], tM) * ct.cdiv(C.shape[5], tN), 1, 1)

    ct.launch(torch.cuda.current_stream(),
              grid,
              tile_contraction_kl,
              (A, B, C, tM, tN, tK))


def launch_tile_contraction_klb(A, B, C, tM, tN, tK):
    # eacxz
    grid = (C.shape[0] * C.shape[1] * C.shape[3] * ct.cdiv(C.shape[4], tM) * ct.cdiv(C.shape[5], tN), 1, 1)

    ct.launch(torch.cuda.current_stream(),
              grid,
              tile_contraction_klb,
              (A, B, C, tM, tN, tK))


def benchmark():
    configs = [
        # (name,                     e, a, b, c, k, l,  x,  y,  z)
        ("1b is expected winner",    4, 4, 4, 4, 4, 4, 32, 32, 32),
        ("1c is expected winner",    4, 4, 4, 4, 4, 4, 32, 32, 32),
    ]
    for name, e, a, b, c, k, l, x, y, z in configs:
        A = torch.rand((e, a, b, k, l, x, y), dtype=torch.float16, device="cuda")
        B = torch.rand((e, c, k, l, y, z),    dtype=torch.float16, device="cuda")
        C = torch.zeros((e, a, b, c, x, z),   dtype=torch.float32, device="cuda")
        tM = next_power_of_two(x // 2)
        tN = next_power_of_two(z // 2)
        tK = next_power_of_two(y // 2)
        grid_1b = e*a*b*c * ct.cdiv(x,tM) * ct.cdiv(z,tN)
        grid_1c = e*a*c   * ct.cdiv(x,tM) * ct.cdiv(z,tN)

        t_1b = triton.testing.do_bench(lambda: launch_tile_contraction_kl( A, B, C, tM, tN, tK))
        t_1c = triton.testing.do_bench(lambda: launch_tile_contraction_klb(A, B, C, tM, tN, tK))
        winner = "1b" if t_1b < t_1c else "1c"
        print(f"[{name}]")
        print(f"  grid: 1b={grid_1b} blocks  1c={grid_1c} blocks")
        print(f"  1b={t_1b:.3f}ms  1c={t_1c:.3f}ms  --> {winner} wins")


def main():
    A, B, C = tensor_initialization()
    
    # Initialize tiles sizes
    tM = next_power_of_two(A.shape[5] // 2)
    tN = next_power_of_two(B.shape[5] // 2)
    tK = next_power_of_two(B.shape[4] // 2)
    print(f"Matrix shapes: A={A.size()}, B={B.size()}, C={C.size()}")
    print(f"Tile sizes: tM={tM}, tN={tN}, tK={tK}")

    C_torch = torch.einsum('eabklxy, ecklyz -> eabcxz', A, B)

    # Task 1b
    C.zero_()
    launch_tile_contraction_kl(A, B, C, tM, tN, tK)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Task 1b failed!"

    # Task 1c
    C.zero_()
    launch_tile_contraction_klb(A, B, C, tM, tN, tK)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Task 1c failed!"

    benchmark()

    return


if __name__ == "__main__":
    main()
