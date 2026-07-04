import torch.nn.functional as F


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
