import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def gn_channel_partials(x, partial_sum, partial_sum_sq,
                        height: ConstInt, width: ConstInt):
    """
    Per (sample, channel): partial sum and sum-of-squares over the channel's H*W.
    """
    num_channels = x.shape[1]
    spatial_size = height * width
    bid = ct.bid(0)
    channel_idx = bid % num_channels
    sample_idx  = bid // num_channels

    t = ct.reshape(ct.load(x, index=(sample_idx, channel_idx, 0, 0),
                           shape=(1, 1, height, width)).astype(torch.float32),
                   (spatial_size,))
    ct.store(partial_sum,    index=(sample_idx, channel_idx), tile=ct.reshape(ct.sum(t,     axis=0), (1, 1)))
    ct.store(partial_sum_sq, index=(sample_idx, channel_idx), tile=ct.reshape(ct.sum(t * t, axis=0), (1, 1)))


@ct.kernel
def gn_apply_from_partials(x, out, weight, bias, partial_sum, partial_sum_sq,
                           channels_per_group: ConstInt,
                           height: ConstInt, width: ConstInt, eps):
    """
    Per (sample, channel): rebuild group stats from the Cg partials, then normalize+affine+SiLU.
    """
    num_channels = x.shape[1]
    spatial_size = height * width
    group_size = channels_per_group * spatial_size

    bid = ct.bid(0)
    channel_idx = bid % num_channels
    sample_idx  = bid // num_channels
    group_idx   = channel_idx // channels_per_group

    # rebuild this channel's group mean / inv_std from the Cg per-channel partials
    group_sum    = ct.zeros((1,), dtype=torch.float32)
    group_sum_sq = ct.zeros((1,), dtype=torch.float32)
    for channel_in_group in range(channels_per_group):
        c = group_idx * channels_per_group + channel_in_group
        group_sum    = group_sum    + ct.reshape(ct.load(partial_sum,    index=(sample_idx, c), shape=(1, 1)), (1,))
        group_sum_sq = group_sum_sq + ct.reshape(ct.load(partial_sum_sq, index=(sample_idx, c), shape=(1, 1)), (1,))

    mean    = group_sum / group_size
    inv_std = ct.rsqrt(group_sum_sq / group_size - mean * mean + eps)

    w = ct.load(weight, index=(channel_idx,), shape=(1,)).astype(torch.float32)
    b = ct.load(bias,   index=(channel_idx,), shape=(1,)).astype(torch.float32)
    t = ct.reshape(ct.load(x, index=(sample_idx, channel_idx, 0, 0),
                           shape=(1, 1, height, width)).astype(torch.float32),
                   (spatial_size,))
    normed = (t - mean) * inv_std
    affine = normed * w + b
    silu   = affine * (1.0 / (1.0 + ct.exp(-affine)))
    ct.store(out, index=(sample_idx, channel_idx, 0, 0),
             tile=ct.reshape(silu.astype(out.dtype), (1, 1, height, width)))


def launch_split_config_kernel(x, weight, bias, num_groups, eps):
    """
    Drop-in alternative to launch_reference_config_kernel. Both passes run N*C blocks.
    """
    N, C, H, W = x.shape
    channels_per_group = C // num_groups
    out            = torch.empty_like(x)
    partial_sum    = torch.empty((N, C), dtype=torch.float32, device=x.device)
    partial_sum_sq = torch.empty((N, C), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream()

    # 320+ blocks -- saturates the SMs
    grid = (N * C, 1, 1)
    ct.launch(stream, grid, gn_channel_partials,
              (x, partial_sum, partial_sum_sq, int(H), int(W)))
    ct.launch(stream, grid, gn_apply_from_partials,
              (x, out, weight, bias, partial_sum, partial_sum_sq,
               channels_per_group, int(H), int(W), float(eps)))
    return out
