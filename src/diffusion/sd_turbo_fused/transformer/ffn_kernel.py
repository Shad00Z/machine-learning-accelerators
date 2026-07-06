import cuda.tile as ct
import torch
ConstInt = ct.Constant[int]

DEFAULT_FFN_TILE = (64, 64, 64)
_FFN_TILE_BY_DIM = {320: (128, 64, 64), 640: (64, 128, 64), 1280: (64, 128, 64)}

def best_ffn_tile(dim, tokens, inner):
    """
    Pick the tuned (tM, tN, tK) for this FFN shape; 
    Fall back to (64, 64, 64).
    """
    tM, tN, tK = _FFN_TILE_BY_DIM.get(dim, DEFAULT_FFN_TILE)
    if tokens % tM or inner % tN or dim % tN:
        return DEFAULT_FFN_TILE
    return (tM, tN, tK)


# ---- Kernel A: LayerNorm first touch -> mm1 (dim -> 2*inner) -> GEGLU last touch -> gated[M, inner] ----
@ct.kernel
def ffn_mm1_geglu(x, w1t, b1, ln_weight, ln_bias, gated,
                  tM: ConstInt, tN: ConstInt, tK: ConstInt, eps):
    """gated[M, inner] = GEGLU( layer_norm(x) @ w1t + b1 ).

    proj = LN(x) @ w1t + b1 has Nfull = 2*inner columns;
    GEGLU splits into (hidden, gate) halves -> hidden * gelu(gate). 
    Fused: two accumulators (hidden cols n_tile, gate cols n_tile + num_tiles_n)
    """
    K     = x.shape[1]
    inner = gated.shape[1]
    num_tiles_n = ct.cdiv(inner, tN)

    bid = ct.bid(0)
    n_tile = bid % num_tiles_n
    m_tile = bid // num_tiles_n

    # LayerNorm: per-row mean / inv_std over K
    row_sum = ct.zeros((tM,), dtype=torch.float32)
    row_sq  = ct.zeros((tM,), dtype=torch.float32)
    for k in range(ct.cdiv(K, tK)):
        x_tile = ct.load(x, index=(m_tile, k), shape=(tM, tK), padding_mode=ct.PaddingMode.ZERO).astype(torch.float32)
        row_sum = row_sum + ct.sum(x_tile,          axis=1)
        row_sq  = row_sq  + ct.sum(x_tile * x_tile, axis=1)
    mean_1d = row_sum / K
    mean    = ct.reshape(mean_1d, (tM, 1))
    inv_std = ct.reshape(ct.rsqrt(row_sq / K - mean_1d * mean_1d + eps), (tM, 1))

    # mm1 for the hidden half (w1t col-tile n_tile) and gate half (col-tile n_tile + num_tiles_n)
    acc_a = ct.zeros((tM, tN), dtype=torch.float32)
    acc_b = ct.zeros((tM, tN), dtype=torch.float32)
    for k in range(ct.cdiv(K, tK)):
        x_tile = ct.load(x, index=(m_tile, k), shape=(tM, tK), padding_mode=ct.PaddingMode.ZERO).astype(torch.float32)
        lw = ct.load(ln_weight, index=(k,), shape=(tK,), padding_mode=ct.PaddingMode.ZERO).astype(torch.float32)
        lb = ct.load(ln_bias,   index=(k,), shape=(tK,), padding_mode=ct.PaddingMode.ZERO).astype(torch.float32)
        normed = ((x_tile - mean) * inv_std * lw + lb).astype(torch.float16)
        # Hidden
        wa = ct.load(w1t, index=(k, n_tile),           shape=(tK, tN), padding_mode=ct.PaddingMode.ZERO)
        acc_a = ct.mma(normed, wa, acc_a)
        
        # Gate
        wb = ct.load(w1t, index=(k, n_tile + num_tiles_n), shape=(tK, tN), padding_mode=ct.PaddingMode.ZERO)
        acc_b = ct.mma(normed, wb, acc_b)

    # GEGLU: hidden * gelu(gate), gelu via tanh approximation
    hidden = acc_a + ct.load(b1, index=(n_tile,),               shape=(tN,)).astype(torch.float32)
    gate = acc_b + ct.load(b1, index=(n_tile + num_tiles_n,), shape=(tN,)).astype(torch.float32)
    gelu_gate = 0.5 * gate * (1.0 + ct.tanh(0.7978845608 * (gate + 0.044715 * gate * gate * gate)))
    gated_tile = hidden * gelu_gate
    ct.store(gated, index=(m_tile, n_tile), tile=gated_tile.astype(gated.dtype))
    return


# ---- Kernel B: mm2 (inner -> dim) + bias -> out[M, dim] ----
@ct.kernel
def ffn_mm2(gated, w2t, b2, out, tM: ConstInt, tN: ConstInt, tK: ConstInt):
    """out[M, dim] = gated[M, inner] @ w2t[inner, dim] + b2[dim].

    w2t = w2.T (inner, dim) so the B tile loads directly as (tK, tN).
    """
    # inner (contraction axis)
    K   = gated.shape[1]
    dim = out.shape[1]

    n_tiles = ct.cdiv(dim, tN)
    bid = ct.bid(0)
    # tN-block of output columns (dim)
    n_tile = bid % n_tiles
    # tM-block of rows
    m_tile = bid // n_tiles

    acc = ct.zeros((tM, tN), dtype=torch.float32)
    for k in range(ct.cdiv(K, tK)):
        a  = ct.load(gated, index=(m_tile, k), shape=(tM, tK), padding_mode=ct.PaddingMode.ZERO)   # index is TILE units (offset = index*shape)
        wb = ct.load(w2t,   index=(k, n_tile), shape=(tK, tN), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, wb, acc)

    b2_tile = ct.load(b2, index=(n_tile,), shape=(tN,)).astype(torch.float32)
    out_tile = acc + b2_tile
    ct.store(out, index=(m_tile, n_tile), tile=out_tile.astype(out.dtype))
    return


def launch_ffn_kernel(x, ln_weight, ln_bias, eps, w1, b1, w2, b2, tM=None, tN=None, tK=None):
    """Full fused FFN: LayerNorm -> mm1 -> GEGLU (Kernel A) -> mm2 (Kernel B).

    x: (B, T, dim). Returns (B, T, dim).
    """
    # 1, 4096, 320
    B, T, dim = x.shape
    M = B * T
    # 2 * inner
    N_cols = w1.shape[0]
    inner = N_cols // 2
    if tM is None or tN is None or tK is None:
        tM, tN, tK = best_ffn_tile(dim, M, inner)

    # Pass 1
    flatten_2d = x.reshape(M, dim)
    w1_transpose = w1.t().contiguous()
    gated = torch.empty((M, inner), dtype=x.dtype, device=x.device)

    gridA = (((M + tM - 1) // tM) * ((inner + tN - 1) // tN), 1, 1)
    ct.launch(torch.cuda.current_stream(), gridA, ffn_mm1_geglu,
              (flatten_2d, w1_transpose, b1, ln_weight, ln_bias, gated, tM, tN, tK, float(eps)))
    
    # Pass 2
    w2_transpose = w2.t().contiguous()
    out2d = torch.empty((M, dim),   dtype=x.dtype, device=x.device)

    gridB = (((M + tM - 1) // tM) * ((dim + tN - 1) // tN), 1, 1)
    ct.launch(torch.cuda.current_stream(), gridB, ffn_mm2, 
              (gated, w2_transpose, b2, out2d, tM, tN, tK))

    return out2d.reshape(B, T, dim)


# Verification only
def launch_mm2(gated, w2, b2, tile=64):
    """Isolated launch of mm2. Verification against F.linear(gated, w2, b2)."""
    M, inner = gated.shape
    dim = w2.shape[0]
    # (inner, dim) -> B in [K, N] layout
    w2t = w2.t().contiguous()
    out = torch.empty((M, dim), dtype=gated.dtype, device=gated.device)
    tM = tN = tK = tile
    grid = (((M + tM - 1) // tM) * ((dim + tN - 1) // tN), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, ffn_mm2, (gated, w2t, b2, out, tM, tN, tK))
    return out


def launch_mm1_geglu(x2d, ln_weight, ln_bias, eps, w1, b1, tile=64):
    """Isolated launch of Kernel A (LN -> mm1 -> GEGLU) -> gated[M, inner]."""
    M, dim = x2d.shape
    # 2 * inner
    Nfull = w1.shape[0]
    inner = Nfull // 2
    # (dim, Nfull)
    w1t = w1.t().contiguous()
    gated = torch.empty((M, inner), dtype=x2d.dtype, device=x2d.device)
    tM = tN = tK = tile
    
    grid = (((M + tM - 1) // tM) * ((inner + tN - 1) // tN), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, ffn_mm1_geglu,
              (x2d, w1t, b1, ln_weight, ln_bias, gated, tM, tN, tK, float(eps)))
    return gated
