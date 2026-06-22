from baseline.image_to_text import initialize_pipeline, model_information
from baseline.resnet_block import model_deconstruction
from utils.roofline import roofline


def main():
    pipe = initialize_pipeline()

    # # prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    # # prompt = "A Star Wars Storm Trooper holding a light saber"
    # # generation(pipeline=pipe, prompt=prompt)

    unet = model_deconstruction(pipe)
    model_information(unet)

    # roofline()
    return


if __name__ == "__main__":
    main()
