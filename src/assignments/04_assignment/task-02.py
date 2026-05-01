import cuda.tile as ct
import torch
import triton

from utils import next_power_of_two, tensor_initialization

ConstInt = ct.Constant[int]

# ===========================================================================
# Task 2a: Fused Elementwise Multiplication Kernel
# ===========================================================================

@ct.kernel
def fused_elem_mult_kernel(A, B, C, D, tM: ConstInt, tN: ConstInt, tK: ConstInt):
    # eabklxy, ecklyz -> eabcxz
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
            
    # Fused elementwise multiplication with D
    tile_D = ct.load(D, index=(e, a, b, c, x, z), shape=(1, 1, 1, 1, tM, tN), padding_mode=ct.PaddingMode.ZERO)
    r_tile_D = ct.reshape(tile_D, (tM, tN))
    acc = acc * r_tile_D

    o_acc = ct.reshape(acc.astype(C.dtype), (1, 1, 1, 1, tM, tN))
    ct.store(C, index=(e, a, b, c, x, z), tile=o_acc)
    return

 
def launch_fused_elem_mult_kernel(A, B, C, D, tM, tN, tK):
    # eabcxz
    grid = (C.shape[0] * C.shape[1] * C.shape[2] * C.shape[3] * ct.cdiv(C.shape[4], tM) * ct.cdiv(C.shape[5], tN), 1, 1)

    ct.launch(torch.cuda.current_stream(),
              grid,
              fused_elem_mult_kernel,
              (A, B, C, D, tM, tN, tK))
    
# ===========================================================================
# Task 2b: Separate Elementwise Multiplication Kernel
# ===========================================================================

@ct.kernel
def elem_mult_kernel(C, D, tM: ConstInt, tN: ConstInt):
    # eabcxz, eabcxz -> eabcxz
    bid = ct.bid(0)

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

    tile_C = ct.load(C, index=(e, a, b, c, x, z), shape=(1, 1, 1, 1, tM, tN), padding_mode=ct.PaddingMode.ZERO)
    tile_D = ct.load(D, index=(e, a, b, c, x, z), shape=(1, 1, 1, 1, tM, tN), padding_mode=ct.PaddingMode.ZERO)

    r_tile_C = ct.reshape(tile_C, (tM, tN))
    r_tile_D = ct.reshape(tile_D, (tM, tN))

    result = r_tile_C * r_tile_D

    ct.store(C, index=(e, a, b, c, x, z), tile=ct.reshape(result, (1, 1, 1, 1, tM, tN)))
    return
    
def launch_elem_mult_kernel(C, D, tM, tN):
    # eabcxz
    grid = (C.shape[0] * C.shape[1] * C.shape[2] * C.shape[3] * ct.cdiv(C.shape[4], tM) * ct.cdiv(C.shape[5], tN), 1, 1)

    ct.launch(torch.cuda.current_stream(),
              grid,
              elem_mult_kernel,
              (C, D, tM, tN))

# Contraction kernel from Task 1
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
# Shared Code
# ===========================================================================

def main():
    
    # Matrix Mult with dims 2048: 2 * 2048^3 = 17,179,869,184 FLOPs
    # Contraction FLOPs: 2 * (e*a*b*c*x*z) * (k*l*y)
    # For example: 2 * (4*4*4*8*64*128) * (8*8*32) = 17,179,869,184 FLOPs
    e, a, b, c, k, l, x, y, z = 4, 4, 4, 8, 8, 8, 64, 32, 128
    
    A = torch.rand((e, a, b, k, l, x, y), dtype=torch.float16, device="cuda")
    B = torch.rand((e, c, k, l, y, z), dtype=torch.float16, device="cuda")
    C = torch.zeros((e, a, b, c , x, z), dtype=torch.float32, device="cuda")
    D = torch.rand_like(C)

    # Initialize tile sizes
    tM = next_power_of_two(A.shape[5] // 2)
    tN = next_power_of_two(B.shape[5] // 2)
    tK = next_power_of_two(B.shape[4] // 2)
    print(f"Shapes: A={tuple(A.shape)}, B={tuple(B.shape)}, C={tuple(C.shape)}")
    print(f"Tiles : tM={tM}, tN={tN}, tK={tK}")

    # Verification
    C_torch = torch.einsum('eabklxy,ecklyz->eabcxz', A.float(), B.float()) * D

    # Task 2a
    C.zero_()
    launch_fused_elem_mult_kernel(A, B, C, D, tM, tN, tK)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Task 2a failed!"
    print("Kernel 2a passed!")

    # Task 2b
    C.zero_()
    launch_tile_contraction_kl(A, B, C, tM, tN, tK)
    launch_elem_mult_kernel(C, D, tM, tN)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Task 2b failed!"
    print("Kernel 2b passed!")

    # Benchmark
    warmup, rep = 100, 1000
    t_fused = triton.testing.do_bench(
        lambda: launch_fused_elem_mult_kernel(A, B, C, D, tM, tN, tK),
        warmup=warmup, rep=rep)

    def no_fusion():
        launch_tile_contraction_kl(A, B, C, tM, tN, tK)
        launch_elem_mult_kernel(C, D, tM, tN)

    t_no_fusion = triton.testing.do_bench(no_fusion, warmup=warmup, rep=rep)

    print(f"\nBenchmark:")
    print(f"    Fused   : {t_fused:.3f} ms")
    print(f"    Separate: {t_no_fusion:.3f} ms")
    print(f"    Speedup : {t_no_fusion / t_fused:.2f}x")


if __name__ == "__main__":
    main()