import collections
import functools
import torch

from diffusers import AutoPipelineForText2Image, Transformer2DModel
from diffusers.models.resnet import ResnetBlock2D
from pathlib import Path
from sd_turbo.resnet.resnet_block import inspect_block
from utils.helper import is_shape_fusable


@functools.lru_cache(maxsize=1)
def initialize_pipeline():
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    return pipe


def generation(pipeline, prompt, seed=0, out_dir="diffusion/outputs"):
    # Reproduce image
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, generator=generator).images[0]

    # Save image
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"sdturbo_seed{seed}.png"
    image.save(path)
    print("saved", path)
    return image


def model_information(unet):
    # Extract resnet Blocks
    resnet_blocks = [m for m in unet.modules() if isinstance(m, ResnetBlock2D)]
    print(len(resnet_blocks))

    transformer_blocks = [m for m in unet.modules() if isinstance(m, Transformer2DModel)]
    print(len(transformer_blocks))

    # Print resnet block information
    print("RESNET BLOCK")
    inspect_block(resnet_blocks[0])
    print("Norm 1: ", resnet_blocks[0].norm1)
    print("SiLU: ",resnet_blocks[0].nonlinearity)
    print("Conv1: ",resnet_blocks[0].conv1)
    print("Conv2: ",resnet_blocks[0].conv2)
    print("Time emb proj: ",resnet_blocks[0].time_emb_proj)

    # Print transformer block information
    print("TRANSFORMER BLOCK")
    inspect_block(transformer_blocks[0])


def survey_groupnorm_shapes(pipe, prompt="a photo", seed=0):
    """
    Run one generation and tally the input shape every ResnetBlock2D GroupNorm sees.
    """
    unet = pipe.unet
    device = next(unet.parameters()).device
    records, handles = [], []

    def make_hook(num_groups):
        # forward_pre_hook: args = (input,)
        def hook(module, args):
            _, c, h, w = args[0].shape
            records.append((c, h, w, num_groups))
        return hook

    for _, block in unet.named_modules():
        if isinstance(block, ResnetBlock2D):
            for which in ("norm1", "norm2"):
                norm = getattr(block, which)
                handles.append(norm.register_forward_pre_hook(
                    make_hook(getattr(norm, "num_groups", None))))

    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, generator=gen)
    for h in handles:
        h.remove()

    counts = collections.Counter(records)

    print(f"{'chan':>5} {'H':>4} {'W':>4} {'grps':>5} {'Cg':>4} {'count':>6}  {'fusable':>8}")
    print("-" * 44)
    fusable = total = 0
    for (c, h, w, g), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        ok = is_shape_fusable(c, h, w, g)
        total += n
        fusable += n if ok else 0
        print(f"{c:>5} {h:>4} {w:>4} {g:>5} {c // g:>4} {n:>6}  {'YES' if ok else '':>8}")
    print("-" * 44)
    print(f"total GN calls: {total}   fusable: {fusable} ({100 * fusable / total:.0f}%)")
    print(f"distinct shapes: {len(counts)}")
    return counts
