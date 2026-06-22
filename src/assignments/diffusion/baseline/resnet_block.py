
def model_deconstruction(pipe):
    unet = pipe.unet
    return unet


def inspect_block(block):
    """
    Prints the structure of a block.

    @param unet: takes in a block to be inspected
    """
    print("------------------------------")
    print(block)
    print("------------------------------")
    return
