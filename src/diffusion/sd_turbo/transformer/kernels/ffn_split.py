"""Register-reduced FFN variant (experiment): split the GEGLU into two single-accumulator passes.

Similar approach as ffn_gemm_split_kernel, but splits mm1 into two halfes:

  ffn_mm1_hidden   -- LayerNorm -> mm1 (over hidden[M, inner])
  ffn_gate_geglu   -- LayerNorm -> mm1 (over  gated[M, inner])
  ffn_mm2          -- reused from ffn_kernel
"""
import cuda.tile as ct
import torch
import torch.nn.functional as F

from sd_turbo.transformer.kernels.ffn import ffn_mm2

ConstInt = ct.Constant[int]
ZERO = ct.PaddingMode.ZERO


@ct.kernel
def ffn_mm1_hidden(x, w1t, b1, ln_weight, ln_bias, hidden,
                   tM: ConstInt, tN: ConstInt, tK: ConstInt, eps):
    """hidden[M, inner] = layer_norm(x) @ w1t[:, :inner] + b1[:inner]  -- one accumulator."""
    K     = x.shape[1]
    inner = hidden.shape[1]
    n_tiles = ct.cdiv(inner, tN)
    bid = ct.bid(0)
    n_tile = bid % n_tiles
    m_tile = bid // n_tiles

    row_sum = ct.zeros((tM,), dtype=torch.float32)
    row_sq  = ct.zeros((tM,), dtype=torch.float32)
    for k in range(ct.cdiv(K, tK)):
        xt = ct.load(x, index=(m_tile, k), shape=(tM, tK), padding_mode=ZERO).astype(torch.float32)
        row_sum = row_sum + ct.sum(xt,      axis=1)
        row_sq  = row_sq  + ct.sum(xt * xt, axis=1)
    mean_1d = row_sum / K
    mean    = ct.reshape(mean_1d, (tM, 1))
    inv_std = ct.reshape(ct.rsqrt(row_sq / K - mean_1d * mean_1d + eps), (tM, 1))

    acc = ct.zeros((tM, tN), dtype=torch.float32)
    for k in range(ct.cdiv(K, tK)):
        xt = ct.load(x, index=(m_tile, k), shape=(tM, tK), padding_mode=ZERO).astype(torch.float32)
        lw = ct.load(ln_weight, index=(k,), shape=(tK,), padding_mode=ZERO).astype(torch.float32)
        lb = ct.load(ln_bias,   index=(k,), shape=(tK,), padding_mode=ZERO).astype(torch.float32)
        normed = ((xt - mean) * inv_std * lw + lb).astype(torch.float16)
        # hidden half
        w = ct.load(w1t, index=(k, n_tile), shape=(tK, tN), padding_mode=ZERO)
        acc = ct.mma(normed, w, acc)

    out = acc + ct.load(b1, index=(n_tile,), shape=(tN,)).astype(torch.float32)
    ct.store(hidden, index=(m_tile, n_tile), tile=out.astype(hidden.dtype))
    return


@ct.kernel
def ffn_gate_geglu(x, w1t, b1, ln_weight, ln_bias, hidden, gated,
                   tM: ConstInt, tN: ConstInt, tK: ConstInt, eps):
    """gated[M, inner] = hidden * gelu( layer_norm(x) @ w1t[:, inner:] + b1[inner:] )  -- one accumulator."""
    K     = x.shape[1]
    inner = gated.shape[1]
    n_tiles = ct.cdiv(inner, tN)
    bid = ct.bid(0)
    n_tile = bid % n_tiles
    m_tile = bid // n_tiles

    row_sum = ct.zeros((tM,), dtype=torch.float32)
    row_sq  = ct.zeros((tM,), dtype=torch.float32)
    for k in range(ct.cdiv(K, tK)):
        xt = ct.load(x, index=(m_tile, k), shape=(tM, tK), padding_mode=ZERO).astype(torch.float32)
        row_sum = row_sum + ct.sum(xt,      axis=1)
        row_sq  = row_sq  + ct.sum(xt * xt, axis=1)
    mean_1d = row_sum / K
    mean    = ct.reshape(mean_1d, (tM, 1))
    inv_std = ct.reshape(ct.rsqrt(row_sq / K - mean_1d * mean_1d + eps), (tM, 1))

    acc = ct.zeros((tM, tN), dtype=torch.float32)
    for k in range(ct.cdiv(K, tK)):
        xt = ct.load(x, index=(m_tile, k), shape=(tM, tK), padding_mode=ZERO).astype(torch.float32)
        lw = ct.load(ln_weight, index=(k,), shape=(tK,), padding_mode=ZERO).astype(torch.float32)
        lb = ct.load(ln_bias,   index=(k,), shape=(tK,), padding_mode=ZERO).astype(torch.float32)
        normed = ((xt - mean) * inv_std * lw + lb).astype(torch.float16)
        # gate half
        w = ct.load(w1t, index=(k, n_tile + n_tiles), shape=(tK, tN), padding_mode=ZERO)
        acc = ct.mma(normed, w, acc)

    g = acc + ct.load(b1, index=(n_tile + n_tiles,), shape=(tN,)).astype(torch.float32)
    gelu_g = 0.5 * g * (1.0 + ct.tanh(0.7978845608 * (g + 0.044715 * g * g * g)))
    h = ct.load(hidden, index=(m_tile, n_tile), shape=(tM, tN)).astype(torch.float32)
    ct.store(gated, index=(m_tile, n_tile), tile=(h * gelu_g).astype(gated.dtype))
    return


def launch_ffn_split(x, ln_weight, ln_bias, eps, w1, b1, w2, b2, tM=64, tN=64, tK=64):
    """Full FFN via the split (one-accumulator) path: hidden -> gate/GEGLU -> mm2."""
    B, T, dim = x.shape
    M = B * T
    x2d = x.reshape(M, dim)
    Nfull = w1.shape[0]
    inner = Nfull // 2

    w1t = w1.t().contiguous()
    hidden = torch.empty((M, inner), dtype=x.dtype, device=x.device)
    gated  = torch.empty((M, inner), dtype=x.dtype, device=x.device)
    s = torch.cuda.current_stream()

    gridA = (((M + tM - 1) // tM) * ((inner + tN - 1) // tN), 1, 1)
    ct.launch(s, gridA, ffn_mm1_hidden,
              (x2d, w1t, b1, ln_weight, ln_bias, hidden, tM, tN, tK, float(eps)))
    ct.launch(s, gridA, ffn_gate_geglu,
              (x2d, w1t, b1, ln_weight, ln_bias, hidden, gated, tM, tN, tK, float(eps)))

    # mm2: plain matmul
    out2d = F.linear(gated, w2, b2)
    return out2d.reshape(B, T, dim)
