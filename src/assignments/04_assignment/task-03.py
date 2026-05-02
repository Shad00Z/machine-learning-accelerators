import cuda.tile as ct
import matplotlib.pyplot as plt
import random
import torch
import triton

from utils import next_power_of_two, tensor_initialization

a = 16
b = 16
c = 32
ConstInt = ct.Constant[int]

# ===========================================================================
# Task 3a: Contraction Kernel - ackm, bcnk -> abnm
# ===========================================================================

@ct.kernel
def tile_contraction(A, B, C, tM: ConstInt, tN: ConstInt, tK: ConstInt):
    bid = ct.bid(0)
    
    # Indices C: Right to left (abnm)
    m = bid % ct.cdiv(C.shape[3], tM)
    bid = bid // ct.cdiv(C.shape[3], tM)
    
    n = bid % ct.cdiv(C.shape[2], tN)
    bid = bid // ct.cdiv(C.shape[2], tN)
    
    b = bid % C.shape[1]
    bid = bid // C.shape[1]
    
    a = bid % C.shape[0]
    
    acc = ct.zeros((tN, tM), dtype=torch.float32)
    
    for c in range(A.shape[1]):
        for k in range(ct.cdiv(A.shape[2], tK)):
            tile_A = ct.load(A, index=(a, c, k, m), shape=(1, 1, tK, tM), padding_mode=ct.PaddingMode.ZERO)
            tile_B = ct.load(B, index=(b, c, n, k), shape=(1, 1, tN, tK), padding_mode=ct.PaddingMode.ZERO)
            
            # Reshape due to rank mismatch
            r_tile_A = ct.reshape(tile_A, (tK, tM))
            r_tile_B = ct.reshape(tile_B, (tN, tK))
            
            acc = ct.mma(r_tile_B, r_tile_A, acc)
            
    o_acc = ct.reshape(acc.astype(C.dtype), (1, 1, tN, tM))
    ct.store(C, index=(a, b, n, m), tile=o_acc)
    return


def launch_contraction_kernel(A, B, C, tM, tN, tK):
    # abnm
    grid = (C.shape[0] * C.shape[1] * ct.cdiv(C.shape[2], tN) * ct.cdiv(C.shape[3], tM), 1, 1)

    ct.launch(torch.cuda.current_stream(),
              grid,
              tile_contraction,
              (A, B, C, tM, tN, tK))
    return


def size_sweep(m, n, k):
    warmup, rep = 100, 1000
    
    if n == 0:
        Ns = []
        compute = []
    
        for l_n in range(17, 129):
            print(f"Iteration: {l_n}")
            bench_A = torch.rand((a, c, k, m), dtype=torch.float16, device="cuda")
            bench_B = torch.rand((b, c, l_n, k), dtype=torch.float16, device="cuda")
            bench_C = torch.zeros((a, b, l_n, m), dtype=torch.float16, device="cuda")
            
            # Initialize tile sizes
            bench_tM = next_power_of_two(bench_A.shape[3] // 2)
            bench_tN = next_power_of_two(bench_B.shape[2] // 2)
            bench_tK = next_power_of_two(bench_B.shape[3] // 2)
            
            bench_ms = triton.testing.do_bench(
                lambda: launch_contraction_kernel(bench_A, bench_B, bench_C, bench_tM, bench_tN, bench_tK), 
                warmup=warmup, 
                rep=rep
            )
            
            flops = 2 * a * b * l_n * m * c * k
            tflops = (flops / 1e12) / (bench_ms / 1e3)
            
            Ns.append(l_n)
            compute.append(tflops)
            
        plt.figure(figsize=(7, 4))
        plt.plot(Ns, compute, marker="o")
        plt.title("Contraction vs N")
        plt.xlabel("N")
        plt.ylabel("TFLOPS")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("task3_n_bandwidth.png", dpi=160)
    
    elif k == 0:
        Ks = []
        compute = []
        
        for l_k in range(17, 129):
            print(f"Iteration: {l_k}")
            bench_A = torch.rand((a, c, l_k, m), dtype=torch.float16, device="cuda")
            bench_B = torch.rand((b, c, n, l_k), dtype=torch.float16, device="cuda")
            bench_C = torch.zeros((a, b, n, m), dtype=torch.float16, device="cuda")
            
            # Initialize tile sizes
            bench_tM = next_power_of_two(bench_A.shape[3] // 2)
            bench_tN = next_power_of_two(bench_B.shape[2] // 2)
            bench_tK = next_power_of_two(bench_B.shape[3] // 2)
            
            bench_ms = triton.testing.do_bench(
                lambda: launch_contraction_kernel(bench_A, bench_B, bench_C, bench_tM, bench_tN, bench_tK), 
                warmup=warmup, 
                rep=rep
            )
            
            flops = 2 * a * b * n * m * c * l_k
            tflops = (flops / 1e12) / (bench_ms / 1e3)
            
            Ks.append(l_k)
            compute.append(tflops)
        
        plt.figure(figsize=(7, 4))
        plt.plot(Ks, compute, marker="o")
        plt.title("Contraction vs K")
        plt.xlabel("K")
        plt.ylabel("TFLOPS")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("task3_k_bandwidth.png", dpi=160)
    
    

def main():
    m = random.randint(4, 512)
    n = random.randint(4, 512)
    k = random.randint(4, 512)
    
    size_A = a * c * k * m
    size_B = b * c * n * k
    size_C = a * b * n * m
    
    while (size_A + size_B) * 2 + size_C * 4 > 34 * 1024**3:
        m = random.randint(4, m)
        n = random.randint(4, n)
        k = random.randint(4, k)
        
        size_A = a * c * k * m
        size_B = b * c * n * k
        size_C = a * b * n * m
    
    A = torch.rand((a, c, k, m), dtype=torch.float16, device="cuda")
    B = torch.rand((b, c, n, k), dtype=torch.float16, device="cuda")
    C = torch.zeros((a, b, n, m), dtype=torch.float32, device="cuda")

    # Initialize tile sizes
    tM = next_power_of_two(A.shape[3] // 2)
    tN = next_power_of_two(B.shape[2] // 2)
    tK = next_power_of_two(B.shape[3] // 2)
    print(f"Shapes: A={tuple(A.shape)}, B={tuple(B.shape)}, C={tuple(C.shape)}")
    print(f"Tiles : tM={tM}, tN={tN}, tK={tK}")

    # Verification
    C_torch = torch.einsum('ackm,bcnk->abnm', A.float(), B.float())

    # Task 3a
    C.zero_()
    launch_contraction_kernel(A, B, C, tM, tN, tK)
    assert torch.allclose(C, C_torch.to(torch.float32), rtol=1e-2), "Task 3a failed!"
    print("Kernel 3a passed!")
    
    # Benchmark
    size_sweep(64, 0, 64)
    size_sweep(64, 64, 0)

    return
    

if __name__ == "__main__":
    main()
    