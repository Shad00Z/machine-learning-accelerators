"""Single-pass fused GroupNorm + SiLU -- one kernel launch (vs the reference's two).

The reference config (gn_silu_kernel.py) runs two launches: a stats kernel (grid = N*groups) that
writes mean/std to DRAM, then an apply kernel (grid = N*C) that reads them back. This variant fuses
both into a single launch, one block per group: it reduces the group in registers, then re-loads the
group's channels to normalize + affine + SiLU. That removes one kernel launch and the mean/std DRAM
round-trip -- the launch-overhead cost that dominates at batch 1 (the SD-Turbo inference regime).

It still reads x twice (the group is too large to keep resident across the reduction for the top
shape), so it is not a DRAM-traffic win at large batch -- its edge is the removed launch, not fewer
passes. Same math and idioms as the reference kernel, so it matches within the same tolerance.
"""
import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def gn_silu_single_pass_kernel(x,
                               out,
                               weight,
                               bias,
                               channels_per_group: ConstInt,
                               height: ConstInt,
                               width: ConstInt,
                               eps):
    C = x.shape[1]
    num_groups = C // channels_per_group
    spatial_size = height * width

    bid = ct.bid(0)
    group_idx = bid % num_groups
    n = bid // num_groups

    # Pass 1: reduce the whole group (channels_per_group * H * W) to mean / inv_std, kept in registers.
    sum    = ct.zeros((1,), dtype=torch.float32)
    sum_sq = ct.zeros((1,), dtype=torch.float32)
    for channel_in_group in range(channels_per_group):
        channel_idx = group_idx * channels_per_group + channel_in_group
        tile = ct.load(x, index=(n, channel_idx, 0, 0), shape=(1, 1, height, width))
        res_tile = ct.reshape(tile.astype(torch.float32), (spatial_size,))
        sum    = sum    + ct.sum(res_tile,            axis=0)
        sum_sq = sum_sq + ct.sum(res_tile * res_tile, axis=0)

    count    = channels_per_group * height * width
    mean     = sum / count
    var      = sum_sq / count - mean * mean
    inv_std  = ct.rsqrt(var + eps)                    # (1,), broadcast over (HW,) below

    # Pass 2: normalize + affine + SiLU each channel of the group, using the stats still in registers.
    for channel_in_group in range(channels_per_group):
        channel_idx = group_idx * channels_per_group + channel_in_group
        weight_s = ct.load(weight, index=(channel_idx,), shape=(1,)).astype(torch.float32)
        bias_s   = ct.load(bias,   index=(channel_idx,), shape=(1,)).astype(torch.float32)

        x_tile = ct.load(x, index=(n, channel_idx, 0, 0), shape=(1, 1, height, width))
        x_fp32 = ct.reshape(x_tile.astype(torch.float32), (spatial_size,))

        normed = (x_fp32 - mean) * inv_std
        affine = normed * weight_s + bias_s
        silu   = affine * (1.0 / (1.0 + ct.exp(-affine)))

        out_tile = ct.reshape(silu.astype(out.dtype), (1, 1, height, width))
        ct.store(out, index=(n, channel_idx, 0, 0), tile=out_tile)
    return


def launch_single_pass_kernel(x, weight, bias, num_groups, eps):
    """Full GroupNorm + SiLU in one launch (one block per group)."""
    N, C, H, W = x.shape
    channels_per_group = C // num_groups

    out = torch.empty_like(x)
    grid = (N * num_groups, 1, 1)
    ct.launch(torch.cuda.current_stream(),
              grid,
              gn_silu_single_pass_kernel,
              (x, out, weight, bias, channels_per_group, H, W, eps))
    return out
