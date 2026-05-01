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


def launch_tile_contraction_kl(A, B, C, tM, tN, tK):
    # eabcxz
    grid = (C.shape[0] * C.shape[1] * C.shape[2] * C.shape[3] * ct.cdiv(C.shape[4], tM) * ct.cdiv(C.shape[5], tN), 1, 1)

    ct.launch(torch.cuda.current_stream(),
              grid,
              tile_contraction_kl,
              (A, B, C, tM, tN, tK))

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


def launch_tile_contraction_klb(A, B, C, tM, tN, tK):
    # eacxz
    grid = (C.shape[0] * C.shape[1] * C.shape[3] * ct.cdiv(C.shape[4], tM) * ct.cdiv(C.shape[5], tN), 1, 1)

    ct.launch(torch.cuda.current_stream(),
              grid,
              tile_contraction_klb,
              (A, B, C, tM, tN, tK))

# ===========================================================================
# Task 1d: Merge l and y
# ===========================================================================

@ct.kernel
def tile_contraction_xyzl(A: torch.tensor, B, C, tM: ConstInt, tN: ConstInt, tK: ConstInt):
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
        for ly in range(A.shape[5]):
            tile_A = ct.load(A, index=(e, a, b, k, x, ly), shape=(1, 1, 1, 1, tM, tK), padding_mode=ct.PaddingMode.ZERO)
            tile_B = ct.load(B, index=(e, c, k, ly, z),    shape=(1, 1, 1, tK, tN),    padding_mode=ct.PaddingMode.ZERO)
            
            # Reshape due to rank mismatch
            r_tile_A = ct.reshape(tile_A, (tM, tK))
            r_tile_B = ct.reshape(tile_B, (tK, tN))
            
            acc = ct.mma(r_tile_A, r_tile_B, acc)
            
    o_acc = ct.reshape(acc.astype(C.dtype), (1, 1, 1, 1, tM, tN))
    ct.store(C, index=(e, a, b, c, x, z), tile=o_acc)
    return


def launch_tile_contraction_xyzl(A, B, C, tM, tN, tK):
    # eabcxz
    grid = (C.shape[0] * C.shape[1] * C.shape[2] * C.shape[3] * ct.cdiv(C.shape[4], tM) * ct.cdiv(C.shape[5], tN), 1, 1)
    
    # eabklxy -> eabkxly
    p_A = torch.permute(A, (0, 1, 2, 3, 5, 4, 6)).contiguous()
    # Reshape A
    r_A = p_A.reshape(A.shape[0],                # e
                      A.shape[1],                # a
                      A.shape[2],                # b
                      A.shape[3],                # k
                      A.shape[5],                # x
                      A.shape[4] * A.shape[6])   # l * y
    
    # Reshape B
    r_B = B.reshape(B.shape[0],                # e
                    B.shape[1],                # c
                    B.shape[2],                # k
                    B.shape[3] * B.shape[4],   # l * y
                    B.shape[5])                # z

    ct.launch(torch.cuda.current_stream(),
              grid,
              tile_contraction_xyzl,
              (r_A, r_B, C, tM, tN, tK))

# ===========================================================================
# Task 1e: GEMM with exyz
# ===========================================================================

@ct.kernel
def tile_contraction_exyz(A, B, C, tM: ConstInt, tN: ConstInt, tK: ConstInt, tC: ConstInt):
    bid = ct.bid(0)
    
    # Indices C: Right to left (abcexz)
    z = bid % ct.cdiv(C.shape[5], tN)
    bid = bid // ct.cdiv(C.shape[5], tN)
    
    x = bid % ct.cdiv(C.shape[4], tM)
    bid = bid // ct.cdiv(C.shape[4], tM)
    
    e = bid % ct.cdiv(C.shape[3], tC)
    bid = bid // ct.cdiv(C.shape[3], tC)
    
    c = bid % C.shape[2]
    bid = bid // C.shape[2]
    
    b = bid % C.shape[1]
    bid = bid // C.shape[1]
    
    a = bid % C.shape[0]
    
    acc = ct.zeros((tC, tM, tN), dtype=torch.float32)
    
    for k in range(A.shape[2]):
        for l in range(A.shape[3]):
            for y in range(ct.cdiv(A.shape[6], tK)):
                tile_A = ct.load(A, index=(a, b, k, l, e, x, y), shape=(1, 1, 1, 1, tC, tM, tK), padding_mode=ct.PaddingMode.ZERO)
                tile_B = ct.load(B, index=(c, k, l, e, y, z),    shape=(1, 1, 1, tC, tK, tN),    padding_mode=ct.PaddingMode.ZERO)
                
                # Reshape due to rank mismatch
                r_tile_A = ct.reshape(tile_A, (tC, tM, tK))
                r_tile_B = ct.reshape(tile_B, (tC, tK, tN))
                
                acc = ct.mma(r_tile_A, r_tile_B, acc)
            
    o_acc = ct.reshape(acc.astype(C.dtype), (1, 1, 1, tC, tM, tN))
    ct.store(C, index=(a, b, c, e, x, z), tile=o_acc)
    return


def launch_tile_contraction_exyz(A, B, C, tM, tN, tK, tC):
    # A: eabklxy -> abklexy
    p_A = torch.permute(A, (1, 2, 3, 4, 0, 5, 6)).contiguous()
    
    # B: ecklyz -> ckleyz
    p_B = torch.permute(B, (1, 2, 3, 0, 4, 5)).contiguous()
    
    # C: eabcxz -> abcexz
    p_C = torch.permute(C, (1, 2, 3, 0, 4, 5)).contiguous()
    
    # abcexz
    grid = (p_C.shape[0] * p_C.shape[1] * p_C.shape[2] * ct.cdiv(p_C.shape[3], tC) * ct.cdiv(p_C.shape[4], tM) * ct.cdiv(p_C.shape[5], tN), 1, 1)
    
    ct.launch(torch.cuda.current_stream(),
              grid,
              tile_contraction_exyz,
              (p_A, p_B, p_C, tM, tN, tK, tC))
    
    # Permute C again: abcexz -> eabcxz and copy back in-place
    C.copy_(torch.permute(p_C, (3, 0, 1, 2, 4, 5)).contiguous())


# ===========================================================================
# Shared Code
# ===========================================================================

def benchmark():
    # eabcxz
    configs = [
        # (name,                     e, a, b, c, k, l, x, y, z)
        ("expected 1b > 1c > 1d",    8, 8, 16, 8, 8, 16, 16, 16, 16),
        ("expected 1c > 1b",         16, 8, 5, 6, 8, 8, 8, 8, 8),
        ("expected 1d > 1b",         30, 18, 10, 7, 7, 5, 4, 5, 4)
    ]
    warmup = 200
    rep = 2000
    for name, e, a, b, c, k, l, x, y, z in configs:
        A = torch.rand((e, a, b, k, l, x, y), dtype=torch.float16, device="cuda")
        B = torch.rand((e, c, k, l, y, z),    dtype=torch.float16, device="cuda")
        C = torch.zeros((e, a, b, c, x, z),   dtype=torch.float32, device="cuda")
        
        size_A = e * a * b * k * l * x * y
        size_B = e * c * k * l * y * z
        size_C = e * a * b * c * x * z
        memory = (size_A + size_B) * 2 + size_C * 4
        if memory > 34 * 1024**3:
            print(f"Skipping config {name} due to memory constraints.")
            continue
        
        tM = next_power_of_two(x // 2)
        tN = next_power_of_two(z // 2)
        tK = next_power_of_two(y // 2)
        grid_1b = e*a*b*c * ct.cdiv(x,tM) * ct.cdiv(z,tN)
        grid_1c = e*a*c   * ct.cdiv(x,tM) * ct.cdiv(z,tN)
        grid_1d = e*a*b*c * ct.cdiv(x,tM) * ct.cdiv(z,tN)

        t_1b = triton.testing.do_bench(lambda: launch_tile_contraction_kl(A, B, C, tM, tN, tK), warmup=warmup, rep=rep)
        t_1c = triton.testing.do_bench(lambda: launch_tile_contraction_klb(A, B, C, tM, tN, tK), warmup=warmup, rep=rep)
        t_1d = triton.testing.do_bench(lambda: launch_tile_contraction_xyzl(A, B, C, tM, tN, tK), warmup=warmup, rep=rep)
        times = sorted([("1b", t_1b), ("1c", t_1c), ("1d", t_1d)], key=lambda x: x[1])
        ranking = " > ".join([name for name, _ in times])
        print(f"[{name}]")
        print(f"  total memory: {(memory) / (1024**3)}GiB")
        print(f"  grid: 1b={grid_1b} blocks, 1c={grid_1c} blocks, 1d={grid_1d} blocks")
        print(f"  tile: tM={tM}, tN={tN}, tK={tK}")
        print(f"  1b={t_1b:.3f}ms  1c={t_1c:.3f}ms  1d={t_1d:.3f}ms")
        print(f"  Ranking: {ranking}")


def main():
    A, B, C = tensor_initialization()
    
    # Initialize tiles sizes
    tM = next_power_of_two(A.shape[5] // 2)
    tN = next_power_of_two(B.shape[5] // 2)
    tK = next_power_of_two(B.shape[4] // 2)
    tC = next_power_of_two(C.shape[0] // 2)
    print(f"Matrix shapes: A={A.size()}, B={B.size()}, C={C.size()}")
    print(f"Tile sizes: tM={tM}, tN={tN}, tK={tK}")

    # Verification
    C_torch = torch.einsum('eabklxy, ecklyz -> eabcxz', A, B)
    
    # Task 1b
    C.zero_()
    launch_tile_contraction_kl(A, B, C, tM, tN, tK)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Task 1b failed!"
    print("Kernel 1 passed!")
    
    # Task 1c
    C.zero_()
    launch_tile_contraction_klb(A, B, C, tM, tN, tK)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Task 1c failed!"
    print("Kernel 2 passed!")
    
    # Task 1d
    C.zero_()
    launch_tile_contraction_xyzl(A, B, C, tM, tN, tK)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Task 1d failed!"
    print("Kernel 3 passed!")
    
    # Task 1e
    C.zero_()
    launch_tile_contraction_exyz(A, B, C, tM, tN, tK, tC)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Task 1e failed!"
    print("Kernel 4 passed!")
    
    benchmark()
    
    return


if __name__ == "__main__":
    main()
