import collections, torch
import torch.nn.functional as F
import torch.nn.functional as F

from diffusers.models.attention import BasicTransformerBlock, FeedForward
from sd_turbo.pipeline import initialize_pipeline


def ffn_reference(x, ln_weight, ln_bias, eps, w1, b1, w2, b2):
    """
    LayerNorm -> GEGLU FeedForward. Weights are the nn.Linear weights (out, in), as stored.

    GEGLU: ``mm1`` projects to ``2*inner``; split into (hidden, gate); output = hidden * gelu(gate).
    """
    normed = F.layer_norm(x, (x.shape[-1],), ln_weight, ln_bias, eps)
    proj = F.linear(normed, w1, b1)                 # (.., 2*inner)
    hidden, gate = proj.chunk(2, dim=-1)            # each (.., inner)
    gated = hidden * F.gelu(gate)                   # GEGLU
    return F.linear(gated, w2, b2)                  # (.., dim)


def inspect_ffn(unet):
    """Exact op sequence + dims of one FeedForward."""
    btb = next(m for m in unet.modules() if isinstance(m, BasicTransformerBlock))
    
    # LayerNorm(320)
    print("norm3:", btb.norm3)
    for i, layer in enumerate(btb.ff.net):
        # GEGLU(proj 320->2560) / Dropout / Linear(1280->320)
        print(f"ff.net[{i}]:", layer)
        
    # GEGLU input projection
    proj = btb.ff.net[0].proj
    
    # output projection
    lin2 = btb.ff.net[-1]
    print(f"mm1: {proj.in_features} -> {proj.out_features}  (GEGLU -> {proj.out_features//2})")
    print(f"mm2: {lin2.in_features} -> {lin2.out_features}")
    return


def survey_ffn_shapes(pipe, prompt="a photo", seed=0):
    """Tally (tokens, dim) and the mm shapes every FeedForward sees in one generation."""
    unet = pipe.unet
    records, handles = [], []

    def hook(module, args):
        _, toks, dim = args[0].shape
        # mm1 output (= 2 * inner, the GEGLU proj)
        proj_out = module.net[0].proj.out_features
        # gated width / mm2 input
        inner    = module.net[-1].in_features
        records.append((toks, dim, proj_out, inner))

    for m in unet.modules():
        if isinstance(m, FeedForward):
            handles.append(m.register_forward_pre_hook(hook))

    gen = torch.Generator(device=next(unet.parameters()).device).manual_seed(seed)
    with torch.no_grad():
        pipe(prompt, num_inference_steps=1, guidance_scale=0.0, generator=gen)
    for h in handles:
        h.remove()

    print(f"{'tokens':>7} {'dim':>6} {'mm1':>16} {'mm2':>16} {'count':>6}")
    for (toks, dim, proj_out, inner), n in collections.Counter(records).most_common():
        print(f"{toks:7d} {dim:6d} {f'{dim}->{proj_out}':>16} {f'{inner}->{dim}':>16} {n:6d}")
    return


def ln_geglu_reference(x, ln_weight, ln_bias, W1, b1, eps):
    # x: T tokens, K channels
    normalized_shape = (x.shape[-1],)
    # LayerNorm: normalize each token independently over K
    x_ln = F.layer_norm(x, normalized_shape, ln_weight, ln_bias, eps)
    # Project up to 2*N_half -- one matmul gives both halves
    proj = F.linear(x_ln, W1, b1)
    # Split: hidden = content, gate = learned soft mask
    hidden, gate = proj.chunk(2, dim=-1)
    # GEGLU: pass hidden through only where GELU(gate) is active
    return hidden * F.gelu(gate)
