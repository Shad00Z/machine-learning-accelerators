import types
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.resnet import ResnetBlock2D

from sd_turbo_fused.resnet.gn_silu_kernel import launch_reference_config_kernel
from utils.helper import is_shape_fusable

# (C, H, W) shapes where the fused kernel beats eager for sd_turbo resnet block shapes (measured with bench_gn_silu)
_VALID_GN_SHAPES = frozenset({
    (320, 64, 64), (640, 32, 32), (640, 64, 64), (320, 32, 32), (640, 16, 16),
    (960, 32, 32), (960, 64, 64), (1280, 32, 32), (1920, 32, 32),
})


class GnSiluFused(nn.Module):
    """Fused GroupNorm + SiLU.

    For the supported fixed shape (N, 320, 64, 64) the cuTile kernel is used.
    All other shapes fall back to the reference PyTorch implementation.
    """

    def __init__(self, group_norm: nn.GroupNorm) -> None:
        super().__init__()
        self.num_groups = group_norm.num_groups
        self.eps = group_norm.eps
        self.weight = group_norm.weight
        self.bias = group_norm.bias

    def _is_supported(self, x: torch.Tensor) -> bool:
        if not x.is_cuda or x.dtype != torch.float16:
            return False
        _, C, H, W = x.shape
        if not is_shape_fusable(C, H, W, self.num_groups):
            return False
        if _VALID_GN_SHAPES is None:
            return True
        return (C, H, W) in _VALID_GN_SHAPES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_supported(x):
            return launch_reference_config_kernel(
                x, self.weight, self.bias, self.num_groups, self.eps
            )
        # Fallback: reference GroupNorm + SiLU
        return F.silu(
            F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        )

    def forward_gn_only(self, x: torch.Tensor) -> torch.Tensor:
        """Return GroupNorm output without the fused SiLU activation.

        Used by the scale_shift path where SiLU must be applied after
        the scale+shift, not before.
        """
        return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


def _fused_forward(
    self: ResnetBlock2D,
    input_tensor: torch.Tensor,
    temb: torch.Tensor,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Replacement for ResnetBlock2D.forward that uses fused GN+SiLU.

    norm1 and norm2 are expected to already be GnSiluFused instances (set by
    patch_resnet_block). The SiLU after each norm is therefore omitted here.
    """
    if len(args) > 0 or kwargs.get("scale", None) is not None:
        from diffusers.utils import deprecate
        deprecation_message = (
            "The `scale` argument is deprecated and will be ignored. Please remove it, "
            "as passing it will raise an error in the future. `scale` should directly be "
            "passed while calling the underlying pipeline component i.e., via "
            "`cross_attention_kwargs`."
        )
        deprecate("scale", "1.0.0", deprecation_message)

    hidden_states = input_tensor

    # norm1 + SiLU (fused)
    hidden_states = self.norm1(hidden_states)

    if self.upsample is not None:
        if hidden_states.shape[0] >= 64:
            input_tensor = input_tensor.contiguous()
            hidden_states = hidden_states.contiguous()
        input_tensor = self.upsample(input_tensor)
        hidden_states = self.upsample(hidden_states)
    elif self.downsample is not None:
        input_tensor = self.downsample(input_tensor)
        hidden_states = self.downsample(hidden_states)

    hidden_states = self.conv1(hidden_states)

    # time embedding
    if self.time_emb_proj is not None:
        if not self.skip_time_act:
            temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]

    if self.time_embedding_norm == "default":
        if temb is not None:
            hidden_states = hidden_states + temb
        # norm2 + SiLU (fused)
        hidden_states = self.norm2(hidden_states)
    elif self.time_embedding_norm == "scale_shift":
        if temb is None:
            raise ValueError(
                f"`temb` should not be None when `time_embedding_norm` is "
                f"{self.time_embedding_norm}"
            )
        time_scale, time_shift = torch.chunk(temb, 2, dim=1)
        # GN only (no SiLU yet); SiLU must come after scale+shift
        hidden_states = self.norm2.forward_gn_only(hidden_states)
        hidden_states = hidden_states * (1 + time_scale) + time_shift
        hidden_states = self.nonlinearity(hidden_states)
    else:
        # norm2 + SiLU (fused)
        hidden_states = self.norm2(hidden_states)

    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states)

    if self.conv_shortcut is not None:
        if self.training:
            input_tensor = input_tensor.contiguous()
        input_tensor = self.conv_shortcut(input_tensor)

    output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
    return output_tensor


def patch_resnet_block(block: ResnetBlock2D) -> ResnetBlock2D:
    """Replace norm1/norm2 with GnSiluFused and patch the block's forward.

    The block is modified in-place and returned for convenience.
    """
    block.norm1 = GnSiluFused(block.norm1)
    block.norm2 = GnSiluFused(block.norm2)
    if block.time_embedding_norm == "scale_shift":
        warnings.warn(
            "patch_resnet_block: block uses time_embedding_norm='scale_shift'. "
            "The fused GN+SiLU kernel is not used for norm2 in this block; "
            "GN is applied via forward_gn_only() and SiLU is applied after scale+shift.",
            stacklevel=2,
        )
    # Bind the patched forward to this specific instance so that
    # other ResnetBlock2D instances are unaffected.
    block.forward = types.MethodType(_fused_forward, block)
    return block


def patch_unet(unet: nn.Module, verbose: bool = False) -> int:
    """Patch all ResnetBlock2D blocks inside unet in-place.

    Returns the number of patched blocks.
    """
    patched = 0
    for name, module in unet.named_modules():
        if isinstance(module, ResnetBlock2D):
            patch_resnet_block(module)
            patched += 1
            if verbose:
                print(f"  patched: {name}")
    if verbose:
        print(f"Total patched ResnetBlock2D blocks: {patched}")
    return patched
