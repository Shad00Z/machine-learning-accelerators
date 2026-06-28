# from sd_turbo.resnet.resnet_block import load_data
# from verification.test_gn_silu import test_gn_silu_reference
from sd_turbo.resnet.resnet_block import model_deconstruction
from sd_turbo.image_to_text import generation, initialize_pipeline, model_information


def main():
    pipe = initialize_pipeline()

    # # prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    # # prompt = "A Star Wars Storm Trooper holding a light saber"
    # prompt = "Will Smith eating spaghetti"
    # generation(pipeline=pipe, prompt=prompt)

    unet = model_deconstruction(pipe)
    model_information(unet)

    # norm_silu(pipe, prompt)

    # Verification of correctness
    # npz_file = "../../../test/diffusion/data/gn_silu_block0.npz"
    # npz_extract = load_data(npz_file)

    # test_gn_silu_reference(npz_extract)

    # roofline()
    return


if __name__ == "__main__":
    main()
