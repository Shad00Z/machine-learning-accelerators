import argparse

from sd_turbo.resnet.reference import model_deconstruction
from sd_turbo.pipeline import generation, initialize_pipeline, model_information
from sd_turbo.resnet.fused_block import patch_unet
from sd_turbo.transformer.fused_block import patch_unet_ffn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an image with SD-Turbo and the fused cuTile kernels."
    )
    parser.add_argument("--prompt", default="Will Smith eating spaghetti",
                        help="Text prompt for the image.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for generation.")
    parser.add_argument("--out-dir", default="diffusion/outputs",
                        help="Directory to write the generated image to.")
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True,
                        help="Patch the U-Net with the fused GN+SiLU and LN+GEGLU kernels.")
    parser.add_argument("--inspect", action="store_true",
                        help="Print the U-Net structure and layer information after generation.")
    return parser.parse_args()


def main():
    args = parse_args()
    pipe = initialize_pipeline()

    if args.fused:
        n_resnet = patch_unet(pipe.unet, verbose=True)
        n_ffn = patch_unet_ffn(pipe.unet, verbose=True)
        print(f"Patched {n_resnet} ResnetBlock2D blocks with fused GN+SiLU kernel.")
        print(f"Patched {n_ffn} transformer FFN blocks with fused LN+GEGLU kernel.")

    generation(pipeline=pipe, prompt=args.prompt, seed=args.seed, out_dir=args.out_dir)

    if args.inspect:
        unet = model_deconstruction(pipe)
        model_information(unet)


if __name__ == "__main__":
    main()
