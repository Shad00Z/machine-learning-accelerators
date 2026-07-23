"""Dump an FFN reference (LayerNorm -> GEGLU FFN) for the fused cuTile kernel.

Runs one SD-Turbo generation and hooks the first 320-dim transformer block's:
  - norm3 (forward-pre)  -> the FFN chain input x                    (1, tokens, dim)
  - ff    (forward)      -> the chain output y = ff(norm3(x))        (1, tokens, dim)

``LayerNorm -> mm1 -> GEGLU -> mm2`` chain
Writes test/diffusion/data/ffn_block0.npz.

Run on the Spark:
    PYTHONPATH=src/diffusion:test/diffusion python test/diffusion/data/ffn_generation.py
"""
import numpy as np
import torch
from diffusers.models.attention import BasicTransformerBlock

from sd_turbo.pipeline import initialize_pipeline
from data.load_helper import data_path

TARGET_DIM = 320
PROMPT = "will smith eating spaghetti"
SEED = 0


def _half(t):
    return t.detach().to(torch.float16).cpu().numpy()


def dump_ffn_reference():
    pipe = initialize_pipeline()
    unet = pipe.unet

    block = next(
        m for m in unet.modules()
        if isinstance(m, BasicTransformerBlock) and m.norm3.normalized_shape[0] == TARGET_DIM
    )
    # GEGLU input projection: Linear(dim -> 2*inner)
    proj = block.ff.net[0].proj
    # output projection:      Linear(inner -> dim)
    lin2 = block.ff.net[-1]
    print(f"target block: dim={TARGET_DIM}  mm1={proj.in_features}->{proj.out_features}  "
          f"mm2={lin2.in_features}->{lin2.out_features}  eps={block.norm3.eps}")

    captured = {}

    # norm3 input = chain input
    def pre_hook(module, args):
        captured.setdefault("x", _half(args[0]))

    # ff output = chain output
    def out_hook(module, args, output):
        captured.setdefault("y", _half(output))

    h1 = block.norm3.register_forward_pre_hook(pre_hook)
    h2 = block.ff.register_forward_hook(out_hook)

    gen = torch.Generator(device=next(unet.parameters()).device).manual_seed(SEED)
    with torch.no_grad():
        pipe(PROMPT, num_inference_steps=1, guidance_scale=0.0, generator=gen)
    h1.remove()
    h2.remove()

    if "x" not in captured or "y" not in captured:
        raise RuntimeError("hooks did not fire -- no matching block ran")

    out_path = data_path().parent / "ffn_block0.npz"
    np.savez(
        out_path,
        x=captured["x"], y=captured["y"],
        ln_weight=_half(block.norm3.weight), ln_bias=_half(block.norm3.bias),
        eps=np.float32(block.norm3.eps),
        # (2*inner, dim), (2*inner,)
        w1=_half(proj.weight), b1=_half(proj.bias),
        # (dim, inner),   (dim,)
        w2=_half(lin2.weight), b2=_half(lin2.bias),
        dim=np.int64(TARGET_DIM),
    )
    print(f"wrote {out_path}")
    print(f"  x{captured['x'].shape}  y{captured['y'].shape}  "
          f"w1{tuple(proj.weight.shape)}  w2{tuple(lin2.weight.shape)}")
    print(f"  x range [{captured['x'].min():.3f}, {captured['x'].max():.3f}]  "
          f"y range [{captured['y'].min():.3f}, {captured['y'].max():.3f}]")


if __name__ == "__main__":
    dump_ffn_reference()
