"""Patch SD-Turbo transformer blocks to use the fused LayerNorm -> GEGLU FFN cuTile kernel.

Each BasicTransformerBlock computes ``ff_output = self.ff(self.norm3(hidden_states))``. Our kernel
takes the raw input (before norm) and folds the LayerNorm in, so we replace ``norm3`` with Identity
and ``ff`` with FusedFFN.
Therefore, ``ff(norm3(x))`` becomes ``FusedFFN(x)`` = the fused kernel on x.
The block's residual add is untouched. Shapes that are not in _VALID_DIMS fall back to torch LayerNorm + FeedForward.

Uses the L2-swizzled kernel.
"""
import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock

from sd_turbo_fused.transformer.ffn_swizzle_kernel import launch_ffn_swizzle

# Dims where the fused kernel actually beats the eager fallback (measured, bench_ffn.bench_compare)
_VALID_DIMS = frozenset({320})


class FusedFFN(nn.Module):
    """LayerNorm(norm3) + GEGLU FeedForward as one cuTile call; falls back to torch when unsupported."""

    def __init__(self, norm3: nn.LayerNorm, ff: nn.Module):
        super().__init__()
        self.eps = norm3.eps
        self.ln_weight = norm3.weight
        self.ln_bias = norm3.bias
        self.w1 = ff.net[0].proj.weight        # (2*inner, dim)
        self.b1 = ff.net[0].proj.bias
        self.w2 = ff.net[-1].weight            # (dim, inner)
        self.b2 = ff.net[-1].bias
        self._norm3 = norm3                     # fallback path
        self._ff = ff

    def _is_supported(self, x: torch.Tensor) -> bool:
        if not x.is_cuda or x.dtype != torch.float16 or x.dim() != 3:
            return False
        _, tokens, dim = x.shape
        inner = self.w1.shape[0] // 2
        
        # 64^3, which needs tokens/dim/inner divisible by 64 (no output padding).
        divisible = tokens % 64 == 0 and dim % 64 == 0 and inner % 64 == 0
        
        # Validity
        return divisible and dim in _VALID_DIMS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_supported(x):
            return launch_ffn_swizzle(x, self.ln_weight, self.ln_bias, self.eps,
                                      self.w1, self.b1, self.w2, self.b2)
        # fallback: original LayerNorm + FeedForward
        return self._ff(self._norm3(x))


def patch_ffn_block(block: BasicTransformerBlock) -> BasicTransformerBlock:
    """norm3 -> Identity, ff -> FusedFFN, so ff(norm3(x)) evaluates the fused kernel on raw x."""
    if isinstance(block.ff, FusedFFN):
        return block
    
    block.ff = FusedFFN(block.norm3, block.ff)
    block.norm3 = nn.Identity()
    return block


def patch_unet_ffn(unet: nn.Module, verbose: bool = False) -> int:
    """Patch every BasicTransformerBlock FFN in-place. Returns the number patched."""
    blocks = [m for m in unet.modules() if isinstance(m, BasicTransformerBlock)]
    
    for i, block in enumerate(blocks):
        patch_ffn_block(block)
        if verbose:
            print(f"  patched FFN block {i}")
    if verbose:
        print(f"Total patched transformer FFNs: {len(blocks)}")
    return len(blocks)
