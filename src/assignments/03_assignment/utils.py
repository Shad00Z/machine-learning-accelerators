import cuda.tile as ct
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import torch
import triton


def next_power_of_two(n: int) -> int:
	p = 1
	while p < n:
		p *= 2
	return p


def heatmap(m):
    df = pd.read_csv(f"src/assignments/03_assignment/task4_nv_{m}_tile_shapes.csv")
    
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

    file_name = f"task4_{m}_tile_shapes.png"
    plt.savefig(file_name, dpi=160)
    print(f"Saved heatmap to {file_name}")


def tileShapeSweep(m, n, k, kernel, out_file, group_size):
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
                            kernel,
                            (A, B, C, A.shape[0], B.shape[1], tM, tN, tK, group_size)),
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
    