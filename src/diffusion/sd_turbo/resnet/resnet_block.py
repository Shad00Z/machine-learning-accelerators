import torch.nn.functional as F

from diffusers import UNet2DConditionModel


def model_deconstruction(pipe) -> UNet2DConditionModel:
    unet = pipe.unet
    return unet


def gn_silu_reference(x, weight, bias, num_groups, eps):
    normed = F.group_norm(x, num_groups, weight=weight, bias=bias, eps=eps)
    return F.silu(normed)


def inspect_block(block):
    """
    Prints the structure of a block.

    @param unet: takes in a block to be inspected
    """
    print("------------------------------")
    print(block)
    print("------------------------------")
    return
