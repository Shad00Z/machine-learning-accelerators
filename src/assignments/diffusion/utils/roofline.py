import numpy as np


def roofline():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DRAM_BW = 273e9  # byte/s
    L2_BW = 2.0e12   # byte/s
    PEAK_TENSOR_FP16 = 120e12  # FLOP/s

    eager = 12 / 8  # 1.5 FLOP/byte
    fused = 12 / 4  # 3.0 FLOP/byte

    ai = np.logspace(-1, 4, 600)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    ax.loglog(ai, np.minimum(ai * DRAM_BW, PEAK_TENSOR_FP16), color="#185FA5",
              label=f"DRAM roof ({DRAM_BW / 1e9:.0f} GB/s)")
    ax.loglog(ai, np.minimum(ai * L2_BW, PEAK_TENSOR_FP16), "--", color="#1D9E75",
              label=f"L2 roof ({L2_BW / 1e12:.0f} TB/s, assumed)")
    ax.axhline(PEAK_TENSOR_FP16, color="red", ls=":", lw=0.9,
               label="Peak FP16 Performance")

    # memory-bound elementwise points (placed on the DRAM roof)
    for ai_pt, name, c in [(eager, "base: GN->SiLU", "#D85A30"),
                           (fused, "fuse: GN->SiLU", "#993C1D")]:
        ax.plot(ai_pt, ai_pt * DRAM_BW, "o", color=c, zorder=5)
        ax.annotate(name, (ai_pt, ai_pt * DRAM_BW), textcoords="offset points",
                    xytext=(8, -3), fontsize=8, color=c)
    ax.annotate("", xy=(fused, fused * DRAM_BW),
                xytext=(eager, eager * DRAM_BW),
                arrowprops=dict(arrowstyle="->", color="#993C1D", lw=1.4))

    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)")
    ax.set_ylabel("Performance (FLOP / s)")
    ax.set_title("Roofline -- GB10: norm/activation fusion")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig("roofline.svg")
    print("gn_eager=%.2f  gn_fused=%.2f" % (eager, fused))
