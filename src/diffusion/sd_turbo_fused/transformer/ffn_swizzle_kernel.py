"""
Swizzled variant of the fused FFN kernels (L2 block swizzle, GROUP_M=8).
"""
import cuda.tile as ct
import torch

from sd_turbo_fused.transformer.ffn_kernel import best_ffn_tile

ConstInt = ct.Constant[int]
ZERO = ct.PaddingMode.ZERO
GROUP_M = 8


@ct.kernel
def ffn_mm1_geglu_sw(x, w1t, b1, ln_weight, ln_bias, gated,
                     tM: ConstInt, tN: ConstInt, tK: ConstInt, eps):
    """L2-swizzled Kernel A: gated[M, inner] = GEGLU( layer_norm(x) @ w1t + b1 )."""
    K     = x.shape[1]
    inner = gated.shape[1]
    M     = x.shape[0]

    num_m = ct.cdiv(M, tM)
    n_tiles = ct.cdiv(inner, tN)
    num_in_group = GROUP_M * n_tiles
    bid = ct.bid(0)
    group_id = bid // num_in_group
    first_m = group_id * GROUP_M
    group_m = min(num_m - first_m, GROUP_M)
    bid_in_group = bid % num_in_group
    m_tile = first_m + bid_in_group % group_m
    n_tile = bid_in_group // group_m

    row_sum = ct.zeros((tM,), dtype=torch.float32)
    row_sq  = ct.zeros((tM,), dtype=torch.float32)
    for k in range(ct.cdiv(K, tK)):
        xt = ct.load(x, index=(m_tile, k), shape=(tM, tK), padding_mode=ZERO).astype(torch.float32)
        row_sum = row_sum + ct.sum(xt,      axis=1)
        row_sq  = row_sq  + ct.sum(xt * xt, axis=1)
    mean_1d = row_sum / K
    mean    = ct.reshape(mean_1d, (tM, 1))
    inv_std = ct.reshape(ct.rsqrt(row_sq / K - mean_1d * mean_1d + eps), (tM, 1))

    acc_a = ct.zeros((tM, tN), dtype=torch.float32)
    acc_b = ct.zeros((tM, tN), dtype=torch.float32)
    for k in range(ct.cdiv(K, tK)):
        xt = ct.load(x, index=(m_tile, k), shape=(tM, tK), padding_mode=ZERO).astype(torch.float32)
        lw = ct.load(ln_weight, index=(k,), shape=(tK,), padding_mode=ZERO).astype(torch.float32)
        lb = ct.load(ln_bias,   index=(k,), shape=(tK,), padding_mode=ZERO).astype(torch.float32)
        normed = ((xt - mean) * inv_std * lw + lb).astype(torch.float16)
        wa = ct.load(w1t, index=(k, n_tile),           shape=(tK, tN), padding_mode=ZERO)
        wb = ct.load(w1t, index=(k, n_tile + n_tiles), shape=(tK, tN), padding_mode=ZERO)
        acc_a = ct.mma(normed, wa, acc_a)
        acc_b = ct.mma(normed, wb, acc_b)

    a = acc_a + ct.load(b1, index=(n_tile,),           shape=(tN,)).astype(torch.float32)
    g = acc_b + ct.load(b1, index=(n_tile + n_tiles,), shape=(tN,)).astype(torch.float32)
    gelu_g = 0.5 * g * (1.0 + ct.tanh(0.7978845608 * (g + 0.044715 * g * g * g)))
    ct.store(gated, index=(m_tile, n_tile), tile=(a * gelu_g).astype(gated.dtype))
    return


@ct.kernel
def ffn_mm2_sw(gated, w2t, b2, out, tM: ConstInt, tN: ConstInt, tK: ConstInt):
    """L2-swizzled Kernel B: out[M, dim] = gated @ w2t + b2."""
    K   = gated.shape[1]
    dim = out.shape[1]
    M   = gated.shape[0]

    num_m = ct.cdiv(M, tM)
    n_tiles = ct.cdiv(dim, tN)
    num_in_group = GROUP_M * n_tiles
    bid = ct.bid(0)
    group_id = bid // num_in_group
    first_m = group_id * GROUP_M
    group_m = min(num_m - first_m, GROUP_M)
    bid_in_group = bid % num_in_group
    m_tile = first_m + bid_in_group % group_m
    n_tile = bid_in_group // group_m

    acc = ct.zeros((tM, tN), dtype=torch.float32)
    for k in range(ct.cdiv(K, tK)):
        a  = ct.load(gated, index=(m_tile, k), shape=(tM, tK), padding_mode=ZERO)
        wb = ct.load(w2t,   index=(k, n_tile), shape=(tK, tN), padding_mode=ZERO)
        acc = ct.mma(a, wb, acc)

    b2_tile = ct.load(b2, index=(n_tile,), shape=(tN,)).astype(torch.float32)
    ct.store(out, index=(m_tile, n_tile), tile=(acc + b2_tile).astype(out.dtype))
    return


def launch_ffn_swizzle(x, ln_weight, ln_bias, eps, w1, b1, w2, b2, tM=None, tN=None, tK=None):
    """Full fused FFN via the L2-swizzled kernels."""
    B, T, dim = x.shape
    M = B * T
    x2d = x.reshape(M, dim)
    inner = w1.shape[0] // 2
    if tM is None or tN is None or tK is None:
        tM, tN, tK = best_ffn_tile(dim, M, inner)
    w1t = w1.t().contiguous()
    w2t = w2.t().contiguous()
    gated = torch.empty((M, inner), dtype=x.dtype, device=x.device)
    out2d = torch.empty((M, dim),   dtype=x.dtype, device=x.device)
    s = torch.cuda.current_stream()

    gridA = (((M + tM - 1) // tM) * ((inner + tN - 1) // tN), 1, 1)
    ct.launch(s, gridA, ffn_mm1_geglu_sw,
              (x2d, w1t, b1, ln_weight, ln_bias, gated, tM, tN, tK, float(eps)))
    gridB = (((M + tM - 1) // tM) * ((dim + tN - 1) // tN), 1, 1)
    ct.launch(s, gridB, ffn_mm2_sw, (gated, w2t, b2, out2d, tM, tN, tK))
    return out2d.reshape(B, T, dim)
