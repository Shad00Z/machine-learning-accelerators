import os
os.environ.setdefault("TORCH_LOGS", "output_code")

import torch
import torch.nn as nn

# Real shape from the U-Net: (1, 320, 64, 64), 32 groups
N, C, H, W = 1, 320, 64, 64
NUM_GROUPS = 32


class GnSilu(nn.Module):
    """GroupNorm -> SiLU, matching the ResnetBlock2D norm1/norm2 + nonlinearity."""
    def __init__(self, num_channels, num_groups):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.gn(x))


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA (run on the Spark)")

    m = GnSilu(C, NUM_GROUPS).eval().half().cuda()
    x = torch.randn(N, C, H, W, dtype=torch.float16, device="cuda")

    print(f"=== compiling GroupNorm + SiLU  N={N} C={C} H={H} W={W} groups={NUM_GROUPS} ===")
    cm = torch.compile(m, backend="inductor", fullgraph=True)
    y = cm(x)                 # triggers Inductor codegen -> TORCH_LOGS prints the kernels
    ref = m(x)
    print(f"[GN+SiLU] out={tuple(y.shape)}  max_abs_diff={(y - ref).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
