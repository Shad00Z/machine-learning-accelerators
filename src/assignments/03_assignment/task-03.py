import torch
import cuda.tile as ct
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import triton

from utils import heatmap

ConstInt = ct.Constant[int]

# ===========================================================================
# Task 3: Benchmarking the Matrix Multiplication Kernel
# ===========================================================================


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


def squaredMatrixThroughput():
    # Valid Matrix Sizes
    sizes = [(s, s, s) for s in [256, 512, 1024, 2048, 4096, 8192]]
    
    # Initialize Tiles Sizes
    tM = 64
    tN = 64
    tK = 64
    print(f"Tile sizes: tM={tM}, tN={tN}, tK={tK}")
    
    # Benchmarking Setup
    dims = []
    perf = []
    warmup = 200
    rep = 2000
    
    for m, n, k in sizes:
        # Initialize Random Squared Tensors
        A = torch.rand(m, k, dtype=torch.float16, device="cuda")
        B = torch.rand(k, n, dtype=torch.float16, device="cuda")
        C = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    
        # Kernel Setup
        grid = (ct.cdiv(n, tN) * ct.cdiv(m, tM), 1, 1)
        
        time_ms = triton.testing.do_bench(
            lambda: ct.launch(torch.cuda.current_stream(),
                            grid,
                            mma_kernel,
                            (A, B, C, tM, tN, tK)),
            warmup=warmup,
            rep=rep
        )
        
        tflops = (2 * m * n * k) / (time_ms / 1e3) / (1e12)
        
        dims.append(m)
        perf.append(tflops)
        
        print(f"m=n=k={m}: {tflops:.2f} TFLOPS")
        
    plt.figure(figsize=(7, 4))
    plt.plot(dims, perf, marker="o")
    plt.title("Computational Performance vs Matrix Sizes")
    plt.xlabel("M=N=K")
    plt.ylabel("Throughput (TFLOPS)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("task3_matrix_sizes.png", dpi=160)
    

def tileShapeThroughput(m, n, k, out_file):
    # Valid Tiles Sizes
    tile_sizes = list(itertools.product([32, 64, 128], repeat=3))
    
    # Benchmarking Setup
    records = []
    warmup = 200
    rep = 2000
    
    for tM, tN, tK in tile_sizes:
        # Initialize Random Tensors
        A = torch.rand(m, k, dtype=torch.float16, device="cuda")
        B = torch.rand(k, n, dtype=torch.float16, device="cuda")
        C = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    
        # Kernel Setup
        grid = (ct.cdiv(n, tN) * ct.cdiv(m, tM), 1, 1)
        
        time_ms = triton.testing.do_bench(
            lambda: ct.launch(torch.cuda.current_stream(),
                            grid,
                            mma_kernel,
                            (A, B, C, tM, tN, tK)),
            warmup=warmup,
            rep=rep
        )
        
        tflops = (2 * m * n * k) / (time_ms / 1e3) / (1e12)
        
        print(f"m=n=k={m}, tm={tM}, tn={tN}, tk={tK}: {tflops:.2f} TFLOPS")
        
        records.append({
            "dim_size": m,
            "tM": tM,
            "tN": tN,
            "tK": tK,
            "tflops": tflops
        })
        
    df = pd.DataFrame(records)
    df.to_csv(out_file, index=False)
    
    
def heatmap(m, in_file, out_file):
    df = pd.read_csv(in_file)
    
    tK = 64
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
    ax.set_title(f"Tile Shape Throughput (TFLOPS)\nm=n=k={m}, k_tile={tK} fixed")
    
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


def main():
    # Task 3a)
    squaredMatrixThroughput()
    
    # Collecting results for tile shapes(27)
    matrix_shapes = [(s, s, s) for s in [256, 512, 2048]]
    
    # Task 3b)
    for m, n, k in matrix_shapes:
        tileShapeThroughput(m, n, k, f"task3_{m}_tile_shapes.csv")
        heatmap(m, f"src/assignments/03_assignment/resources-03/task3_{m}_tile_shapes.csv" ,f"task3_{m}_tile_shapes.png")
        
    # For Task 4
    tileShapeThroughput(8192, 8192, 4096, "task3_8192_tile_shapes.csv")


if __name__ == "__main__":
    main()
