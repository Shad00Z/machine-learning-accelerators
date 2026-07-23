"""FFN variant: fuse (LayerNorm, mm1) and (GEGLU, mm2) -- one accumulator per kernel.

Motivation: NCU showed the fully-fused Kernel A (ffn_mm1_geglu) is occupancy-bound (255 regs / 12 %)
because the GEGLU keeps two live accumulators. This variant moves the GEGLU to the second matmul:

  Kernel A' = ffn_mm1_ln     -- LayerNorm -> mm1 -> full proj[M, 2*inner]   (one accumulator; reused)
  Kernel B' = ffn_geglu_mm2  -- gate proj in the mm2 A-operand first touch -> mm2  (one accumulator)
"""
import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]
ZERO = ct.PaddingMode.ZERO

@ct.kernel
def ffn_mm1_ln(x, w1t, b1, ln_weight, ln_bias, proj,
               tM: ConstInt, tN: ConstInt, tK: ConstInt, eps):
    """proj[M, Nfull] = layer_norm(x[M, dim]) @ w1t[dim, Nfull] + b1[Nfull].

    LayerNorm normalizes each row over dim (== the contraction axis) with affine
    ln_weight/ln_bias, folded as the matmul prologue. w1t = w1.T (dim, Nfull=2*inner).
    """
    # dim (LayerNorm axis == contraction axis)
    K     = x.shape[1]
    # 2 * inner
    Nfull = proj.shape[1]

    n_tiles = ct.cdiv(Nfull, tN)
    bid = ct.bid(0)
    n_tile = bid % n_tiles
    m_tile = bid // n_tiles

    # LayerNorm: per-row mean / inv_std over K (reduce the tK axis, accumulate over k-tiles)
    row_sum = ct.zeros((tM,), dtype=torch.float32)
    row_sq  = ct.zeros((tM,), dtype=torch.float32)
    for k in range(ct.cdiv(K, tK)):
        xt = ct.load(x, index=(m_tile, k), shape=(tM, tK), padding_mode=ct.PaddingMode.ZERO).astype(torch.float32)
        row_sum = row_sum + ct.sum(xt,      axis=1)
        row_sq  = row_sq  + ct.sum(xt * xt, axis=1)
    mean_1d = row_sum / K
    var_1d  = row_sq / K - mean_1d * mean_1d
    mean    = ct.reshape(mean_1d, (tM, 1))
    inv_std = ct.reshape(ct.rsqrt(var_1d + eps), (tM, 1))

    # mm1 with LN normalization applied to each A tile
    acc = ct.zeros((tM, tN), dtype=torch.float32)
    for k in range(ct.cdiv(K, tK)):
        xt = ct.load(x, index=(m_tile, k), shape=(tM, tK), padding_mode=ct.PaddingMode.ZERO).astype(torch.float32)
        lw = ct.load(ln_weight, index=(k,), shape=(tK,), padding_mode=ct.PaddingMode.ZERO).astype(torch.float32)
        lb = ct.load(ln_bias,   index=(k,), shape=(tK,), padding_mode=ct.PaddingMode.ZERO).astype(torch.float32)
        # (tM,1) & (tK,) broadcast over (tM,tK)
        normed = (xt - mean) * inv_std * lw + lb
        wt = ct.load(w1t, index=(k, n_tile), shape=(tK, tN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(normed.astype(torch.float16), wt, acc)

    b1_tile = ct.load(b1, index=(n_tile,), shape=(tN,)).astype(torch.float32)
    out_tile = acc + b1_tile
    ct.store(proj, index=(m_tile, n_tile), tile=out_tile.astype(proj.dtype))


@ct.kernel
def ffn_geglu_mm2(proj, w2t, b2, out, tM: ConstInt, tN: ConstInt, tK: ConstInt):
    """
    out[M, dim] = GEGLU(proj) @ w2t + b2, with the gate fused into the mm2 A-operand (last touch).

    proj[M, 2*inner] is the mm1 output. 
    The gated A-operand is gated[m, k] = proj[m, k] * gelu(proj[m, inner + k])
    """
    Nfull = proj.shape[1]
    inner = Nfull // 2
    dim   = out.shape[1]

    n_tiles = ct.cdiv(dim, tN)
    bid = ct.bid(0)
    n_tile = bid % n_tiles
    m_tile = bid // n_tiles

    # gate-half column-tile offset in proj
    inner_tiles = ct.cdiv(inner, tK)
    acc = ct.zeros((tM, tN), dtype=torch.float32)
    for k in range(inner_tiles):
        h = ct.load(proj, index=(m_tile, k),               shape=(tM, tK), padding_mode=ZERO).astype(torch.float32)
        g = ct.load(proj, index=(m_tile, k + inner_tiles), shape=(tM, tK), padding_mode=ZERO).astype(torch.float32)
        gelu_g = 0.5 * g * (1.0 + ct.tanh(0.7978845608 * (g + 0.044715 * g * g * g)))
        # A-tile of mm2, gated on the fly
        gated = (h * gelu_g).astype(torch.float16)
        wb = ct.load(w2t, index=(k, n_tile), shape=(tK, tN), padding_mode=ZERO)
        acc = ct.mma(gated, wb, acc)

    b2_tile = ct.load(b2, index=(n_tile,), shape=(tN,)).astype(torch.float32)
    ct.store(out, index=(m_tile, n_tile), tile=(acc + b2_tile).astype(out.dtype))


def launch_ffn_gemm_split(x, ln_weight, ln_bias, eps, w1, b1, w2, b2, tM=64, tN=64, tK=64):
    """Full FFN via the (LN,mm1) + (GEGLU,mm2) split -- one accumulator per kernel."""
    B, T, dim = x.shape
    M = B * T
    x2d = x.reshape(M, dim)
    # 2 * inner
    Nfull = w1.shape[0]
    # (dim, Nfull)
    w1t = w1.t().contiguous()
    # (inner, dim)
    w2t = w2.t().contiguous()
    proj  = torch.empty((M, Nfull), dtype=x.dtype, device=x.device)
    out2d = torch.empty((M, dim),   dtype=x.dtype, device=x.device)
    s = torch.cuda.current_stream()

    # Kernel A': LayerNorm -> mm1 -> full proj  (reused, one accumulator)
    gridA = (((M + tM - 1) // tM) * ((Nfull + tN - 1) // tN), 1, 1)
    ct.launch(s, gridA, ffn_mm1_ln,
              (x2d, w1t, b1, ln_weight, ln_bias, proj, tM, tN, tK, float(eps)))

    # Kernel B': GEGLU (in mm2 last touch) -> mm2  (one accumulator)
    gridB = (((M + tM - 1) // tM) * ((dim + tN - 1) // tN), 1, 1)
    ct.launch(s, gridB, ffn_geglu_mm2, (proj, w2t, b2, out2d, tM, tN, tK))
    return out2d.reshape(B, T, dim)


# For testing purposes
def launch_mm1_ln(x2d, ln_weight, ln_bias, eps, w1, b1, tile=64):
    """Isolated launch of mm1 + LN first touch"""
    M, dim = x2d.shape
    # 2 * inner
    Nfull = w1.shape[0]
    # (dim, Nfull)
    w1t = w1.t().contiguous()
    proj = torch.empty((M, Nfull), dtype=x2d.dtype, device=x2d.device)
    
    tM = tN = tK = tile
    
    grid = (((M + tM - 1) // tM) * ((Nfull + tN - 1) // tN), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, ffn_mm1_ln,
              (x2d, w1t, b1, ln_weight, ln_bias, proj, tM, tN, tK, float(eps)))
    return proj
