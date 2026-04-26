import cuda.tile as ct
import torch
import triton

# ===========================================================================
# Task 1: FP32 vs FP16 Performance
# ===========================================================================

M_TILE_SIZE = 64
N_TILE_SIZE = 64
K_TILE_SIZE = 64


@ct.kernel
def kernel_fp16(A, B, C):
    acc = ct.zeros((M_TILE_SIZE, N_TILE_SIZE), dtype=torch.float32)

    for k in range(4096 // K_TILE_SIZE):  # loop over K-tiles inside the kernel
        tile_A = ct.load(A, index=(0, k), shape=(M_TILE_SIZE, K_TILE_SIZE))
        tile_B = ct.load(B, index=(k, 0), shape=(K_TILE_SIZE, N_TILE_SIZE))
        acc = ct.mma(tile_A, tile_B, acc)

    ct.store(C, index=(0, 0), tile=acc)
    return


@ct.kernel
def kernel_fp32(A, B, C):
    acc = ct.zeros((M_TILE_SIZE, N_TILE_SIZE), dtype=torch.float32)

    for k in range(4096 // K_TILE_SIZE):  # loop over K-tiles inside the kernel
        tile_A = ct.load(A, index=(0, k), shape=(M_TILE_SIZE, K_TILE_SIZE))
        tile_B = ct.load(B, index=(k, 0), shape=(K_TILE_SIZE, N_TILE_SIZE))
        acc = ct.mma(tile_A, tile_B, acc)

    ct.store(C, index=(0, 0), tile=acc)
    return


def main():
    # Reproducibility
    torch.manual_seed(0)

    # FP16: Matrix initialization
    A16 = torch.rand(64, 4096, dtype=torch.float16, device='cuda')
    B16 = torch.rand(4096, 64, dtype=torch.float16, device='cuda')

    # FP32: Matrix initialization
    A32 = torch.rand(64, 4096, dtype=torch.float32, device='cuda')
    B32 = torch.rand(4096, 64, dtype=torch.float32, device='cuda')

    C32 = torch.zeros(64, 64, dtype=torch.float32, device='cuda')

    # Grid
    grid = (1, 1, 1)

    # FP16: Correctness
    ct.launch(torch.cuda.current_stream(),
              grid,
              kernel_fp16,
              (A16, B16, C32))
    C32_torch = torch.matmul(A16, B16)
    assert torch.allclose(C32, C32_torch.to(torch.float32), rtol=1e-2), "FP16 Matmul failed!"

    # FP16: Correctness
    C32 = torch.zeros(64, 64, dtype=torch.float32, device='cuda')

    ct.launch(torch.cuda.current_stream(),
              grid,
              kernel_fp32,
              (A32, B32, C32))
    C32_torch = torch.matmul(A32, B32)
    assert torch.allclose(C32, C32_torch.to(torch.float32), rtol=1e-2), "FP32 Matmul failed!"

    # Benchmarking Setup
    warmup = 200
    rep = 2000

    # FP16: Benchmarking
    ms_fp16 = triton.testing.do_bench(
        lambda: ct.launch(torch.cuda.current_stream(),
                          grid,
                          kernel_fp16,
                          (A16, B16, C32)),
        warmup=warmup,
        rep=rep
    )

    # FP32: Benchmarking
    ms_fp32 = triton.testing.do_bench(
        lambda: ct.launch(torch.cuda.current_stream(),
                          grid,
                          kernel_fp32,
                          (A32, B32, C32)),
        warmup=warmup,
        rep=rep
    )

    # Comparison
    print("Benchmark (ms per launch):")
    print("    FP16 time: ", ms_fp16)
    print("    FP32 time: ", ms_fp32)

    return

if __name__ == "__main__":
    main()
