import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]
# Fixed shapes: N=1, C=320, G=32, H=W=64
#   Cg = C // G   = 10      channels per group
#   HW = H * W    = 4096    elements per channel
#   group_elems   = Cg*HW = 40960  reduction domain per

@ct.kernel
def gn_mean_stddev_kernel(x, 
                          mean, 
                          std_dev, 
                          channels_per_group: ConstInt, 
                          height: ConstInt, 
                          width: ConstInt,
                          eps
    ):
    # 1) Prepare
    num_groups = mean.shape[1]
    bid = ct.bid(0)
    group_idx = bid % num_groups
    n = bid // num_groups

    # 2) Reduction / Accumulation
    sum    = ct.zeros((1,), dtype=torch.float32)
    sum_sq = ct.zeros((1,), dtype=torch.float32)
    for channel_in_group in range(channels_per_group):
        channel_idx = group_idx * channels_per_group + channel_in_group

        tile = ct.load(x, index=(n, channel_idx, 0, 0), shape=(1, 1, height, width))
        res_tile = ct.reshape(tile.astype(torch.float), (height * width,))

        sum = sum + ct.sum(res_tile, axis=0)
        sum_sq = sum_sq + ct.sum(res_tile * res_tile, axis=0)

    # 3) Store
    count = channels_per_group * height * width
    group_mean    = sum / count
    group_var     = sum_sq / count - group_mean * group_mean
    group_std_dev = ct.rsqrt(group_var + eps)

    ct.store(mean, index=(n, group_idx), tile=ct.reshape(group_mean, (1, 1)))
    ct.store(std_dev, index=(n, group_idx), tile=ct.reshape(group_std_dev, (1, 1)))
    return


@ct.kernel
def gn_silu_kernel(x, 
                   out, 
                   weight, 
                   bias, 
                   mean, 
                   std_dev, 
                   channels_per_group: ConstInt, 
                   height: ConstInt, 
                   width: ConstInt):
    spatial_size = height * width
    
    # index calculation
    bid = ct.bid(0)
    channel_idx = bid % x.shape[1]
    sample_idx  = bid // x.shape[1]
    group_idx   = channel_idx // channels_per_group

    # per-group, loaded as 1-element tiles to broadcast over the channel
    mean_s    = ct.reshape(ct.load(mean,    index=(sample_idx, group_idx), shape=(1, 1)), (1,))  # VERIFY
    inv_std_s = ct.reshape(ct.load(std_dev, index=(sample_idx, group_idx), shape=(1, 1)), (1,))  # VERIFY
    weight_s  = ct.load(weight, index=(channel_idx,), shape=(1,)).astype(torch.float32)
    bias_s    = ct.load(bias,   index=(channel_idx,), shape=(1,)).astype(torch.float32)

    x_tile = ct.load(x, index=(sample_idx, channel_idx, 0, 0), shape=(1, 1, height, width))
    x_fp32 = ct.reshape(x_tile.astype(torch.float32), (spatial_size,))
    
    normed = (x_fp32 - mean_s) * inv_std_s            # broadcast (1,) over (HW,)  # VERIFY broadcast
    affine = normed * weight_s + bias_s
    silu   = affine * (1.0 / (1.0 + ct.exp(-affine))) # SiLU, composed             # VERIFY scalar+tile

    out_tile = ct.reshape(silu.astype(out.dtype), (1, 1, height, width))
    ct.store(out, index=(sample_idx, channel_idx, 0, 0), tile=out_tile)
    return


def launch_reference_config_kernel(x, weight, bias, num_groups, eps):
    # 1, 320, 64, 64
    N, C, H, W = x.shape
    # 320 // 32 = 10
    channels_per_group = C  // num_groups

    out     = torch.empty_like(x)
    mean    = torch.empty((N, num_groups), dtype=torch.float32, device=x.device)
    std_dev = torch.empty((N, num_groups), dtype=torch.float32, device=x.device)

    # N = 1, num_groups = 32
    grid1 = (N * num_groups, 1, 1)

    ct.launch(torch.cuda.current_stream(),
              grid1,
              gn_mean_stddev_kernel,
              (x, mean, std_dev, channels_per_group, H, W, eps))

    # N = 1, C = 320
    grid2 = (N * C, 1, 1)

    ct.launch(torch.cuda.current_stream(),
              grid2,
              gn_silu_kernel,
              (x, out, weight, bias, mean, std_dev, channels_per_group, H, W))
    return out
