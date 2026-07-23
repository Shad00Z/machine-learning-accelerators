import os
os.environ.setdefault("TORCH_LOGS", "output_code")

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward

TOKENS, DIM = 4096, 320


class LNFFN(nn.Module):
    """LayerNorm -> GEGLU FeedForward, matching the transformer block's norm3 -> ff."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, activation_fn="geglu")   # dim -> 2*inner (GEGLU) -> inner -> dim

    def forward(self, x):
        return self.ff(self.norm(x))


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA (run on the Spark)")

    m = LNFFN(DIM).eval().half().cuda()
    x = torch.randn(1, TOKENS, DIM, dtype=torch.float16, device="cuda")

    print(f"=== compiling LN + GEGLU FFN  tokens={TOKENS} dim={DIM} ===")
    cm = torch.compile(m, backend="inductor", fullgraph=True)
    y = cm(x)                 # triggers Inductor codegen -> TORCH_LOGS prints the kernels above
    ref = m(x)
    print(f"[LN+GEGLU] out={tuple(y.shape)}  max_abs_diff={(y - ref).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
