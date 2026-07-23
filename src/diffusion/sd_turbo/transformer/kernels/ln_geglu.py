import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

# Fixed shapes: T=77, K=320, N_half=1280
#   tK = 64  (K=320 = 5*tK, exact)
#   tN = 64  (N_half=1280 = 20*tN, exact)

_TK = 64
_TN = 64


@ct.kernel
def ln_stats_kernel(
    x,
    mean,
    rstd,
    K: ConstInt,
    tK: ConstInt,
    eps,
):
    # 1) Prepare
    t = ct.bid(0)

    sum_val = ct.zeros((1,), dtype=torch.float32)
    sum_sq = ct.zeros((1,), dtype=torch.float32)

    # 2) Reduction
    for k in range(K // tK):
        tile = ct.reshape(
            ct.load(x, index=(t, k), shape=(1, tK)).astype(torch.float32),
            (tK,),
        )
        sum_val = sum_val + ct.sum(tile, axis=0)
        sum_sq = sum_sq + ct.sum(tile * tile, axis=0)

    # 3) Store
    mean_val = sum_val / K
    var_val = sum_sq / K - mean_val * mean_val
    rstd_val = ct.rsqrt(var_val + eps)

    ct.store(mean, index=(t,), tile=ct.reshape(mean_val, (1,)))
    ct.store(rstd, index=(t,), tile=ct.reshape(rstd_val, (1,)))
    return


@ct.kernel
def ln_geglu_mm1_kernel(
    x,
    ln_weight,
    ln_bias,
    mean,
    rstd,
    W1_T,  # (K, 2*N_half), W1 transposed
    b1,  # (2*N_half,)
    out,  # (T, N_half)
    K: ConstInt,
    tK: ConstInt,
    tN: ConstInt,
    half_n_tiles: ConstInt,  # N_half // tN = 20
):
    # 1) Prepare
    t = ct.bid(0) // half_n_tiles
    bid_n = ct.bid(0) % half_n_tiles

    mean_s = ct.reshape(ct.load(mean, index=(t,), shape=(1,)), (1,))  # VERIFY
    rstd_s = ct.reshape(ct.load(rstd, index=(t,), shape=(1,)), (1,))  # VERIFY

    acc_gate = ct.zeros((1, tN), dtype=torch.float32)
    acc_hidden = ct.zeros((1, tN), dtype=torch.float32)

    # 2) Accumulation: LN first-touch + matmul
    for k in range(K // tK):
        x_1d = ct.reshape(
            ct.load(x, index=(t, k), shape=(1, tK)).astype(torch.float32),
            (tK,),
        )
        ln_w = ct.load(ln_weight, index=(k,), shape=(tK,)).astype(torch.float32)
        ln_b = ct.load(ln_bias, index=(k,), shape=(tK,)).astype(torch.float32)

        x_ln_1d = (x_1d - mean_s) * rstd_s * ln_w + ln_b  # VERIFY broadcast
        x_ln = ct.reshape(x_ln_1d, (1, tK))

        w_gate = ct.load(W1_T, index=(k, bid_n), shape=(tK, tN)).astype(torch.float32)
        w_hidden = ct.load(
            W1_T, index=(k, half_n_tiles + bid_n), shape=(tK, tN)
        ).astype(torch.float32)

        acc_gate = ct.mma(x_ln, w_gate, acc_gate)
        acc_hidden = ct.mma(x_ln, w_hidden, acc_hidden)

    # 3) Bias + GEGLU last-touch, store
    acc_gate = acc_gate + ct.reshape(
        ct.load(b1, index=(bid_n,), shape=(tN,)).astype(torch.float32), (1, tN)
    )
    acc_hidden = acc_hidden + ct.reshape(
        ct.load(b1, index=(half_n_tiles + bid_n,), shape=(tN,)).astype(torch.float32),
        (1, tN),
    )

    # GEGLU: gate * FastGELU(hidden)
    # see https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    # FastGELU(x) = 0.5 * x * (1 + tanh(0.7978845608 * x * (1 + 0.044715 * x^2)))
    fast_gelu = 0.5 * acc_hidden * (1.0 + ct.tanh(acc_hidden * 0.7978845608 * (1.0 + 0.044715 * acc_hidden * acc_hidden)))
    geglu_out = acc_gate * fast_gelu

    ct.store(out, index=(t, bid_n), tile=geglu_out.astype(out.dtype))
    return


def launch_ln_geglu_mm1(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    T, K = x.shape
    N_half = W1.shape[0] // 2
    tK = _TK
    tN = _TN
    half_n_tiles = N_half // tN

    mean = torch.empty((T,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    out = torch.empty((T, N_half), dtype=x.dtype, device=x.device)

    W1_T = W1.t().contiguous()
    stream = torch.cuda.current_stream()

    ct.launch(stream, (T, 1, 1), ln_stats_kernel, (x, mean, rstd, K, tK, eps))

    ct.launch(
        stream,
        (T * half_n_tiles, 1, 1),
        ln_geglu_mm1_kernel,
        (x, ln_weight, ln_bias, mean, rstd, W1_T, b1, out, K, tK, tN, half_n_tiles),
    )

    return out
