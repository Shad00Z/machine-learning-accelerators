import functools
import torch

from diffusers import AutoPipelineForText2Image, Transformer2DModel
from diffusers.models.resnet import ResnetBlock2D
from pathlib import Path
from sd_turbo.resnet.resnet_block import inspect_block


@functools.lru_cache(maxsize=1)
def initialize_pipeline():
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    return pipe


def generation(pipeline, prompt, seed=0, out_dir="diffusion/outputs"):
    # Reproduce image
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, generator=generator).images[0]

    # Safe image
    Path(out_dir).mkdir(exist_ok=True)
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
