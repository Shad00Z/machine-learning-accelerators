import torch
import cuda.tile as ct
import triton
import matplotlib.pyplot as plt

ConstInt = ct.Constant[int]

# ===========================================================================
# Task 4: Benchmarking Bandwidth
# ===========================================================================

@ct.kernel
def copy_matrix_kernel(A, B, tM: ConstInt, tN: ConstInt):
    m = ct.bid(0)
    n = ct.bid(1)

    tile_a = ct.load(
        A,
        index=(m, n),
        shape=(tM, tN),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(B, index=(m, n), tile=tile_a)
    
def copy_matrix(A, tM, tN):
    num_rows = A.shape[0]
    num_cols = A.shape[1]

    assert tM > 0 and tN > 0, "Tile dimensions must be positive"
    assert (tM & (tM - 1)) == 0 and (tN & (tN - 1)) == 0, "Tile dimensions must be powers of 2"
    
    grid = (ct.cdiv(num_rows, tM), ct.cdiv(num_cols, tN), 1)
    B = torch.empty_like(A)

    ct.launch(torch.cuda.current_stream(),
              grid,
              copy_matrix_kernel,
              (A, B, tM, tN))
    
    return B

def test_copy_matrix():
    M, N = 2**7, 2**7
    tM, tN = 2**4, 2**4
    
    A = torch.randn(M, N, device='cuda')
    B = copy_matrix(A, tM, tN)
    
    assert torch.allclose(A, B), "The copied matrix does not match the original."
    
def bench_copy_matrix():
    M = 2048
    tM = 64
    print("Running bandwidth benchmark for copy_matrix")

    Ns = []
    bandwidths = []

    for N in [16, 32, 64, 128]:
        tN = N
        grid = (ct.cdiv(M, tM), ct.cdiv(N, tN), 1)

        A = torch.randn(M, N, device="cuda")
        B = torch.empty_like(A)

        def launch():
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                copy_matrix_kernel,
                (A, B, tM, tN),
            )

        time_ms = triton.testing.do_bench(launch, warmup=10, rep=100)
        bw = (2 * M * N * A.element_size()) / (time_ms / 1000.0) / (1024 ** 3)  # GB/s

        Ns.append(N)
        bandwidths.append(bw)

        print(f"  M={M:4}, N={N:3}, tM={tM:2}, tN={tN:3}: {bw:.4f} GB/s")

    plt.figure(figsize=(7, 4))
    plt.plot(Ns, bandwidths, marker="o")
    plt.title("copy_matrix bandwidth vs N")
    plt.xlabel("N")
    plt.ylabel("Bandwidth (GB/s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("task4_bandwidth.png", dpi=160)
    
if __name__ == "__main__":
    test_copy_matrix()
    bench_copy_matrix()