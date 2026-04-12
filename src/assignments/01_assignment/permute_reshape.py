import torch


def permute_example(v1: torch.Tensor):
    v2 = v1.permute(1, 0)
    print("------------------------")
    print("PERMUTE")
    print("------------------------")
    print("Tensor after permute:\n", v2)
    print("Tensor shape after permute:", v2.shape)
    print("Tensor stride after permute:", v2.stride())
    print("Tensor contiguity:", v2.is_contiguous(), "\n")


def reshape_example(v1: torch.Tensor):
    v3 = v1.reshape(1, -1)
    print("------------------------")
    print("RESHAPE")
    print("------------------------")
    print("Tensor after reshape:\n", v3)
    print("Tensor shape after reshape:", v3.shape)
    print("Tensor stride after reshape:", v3.stride())
    print("Tensor contiguity:", v3.is_contiguous(), "\n")


def view_example(v1: torch.Tensor):
    v4 = v1.view(1, -1)
    print("------------------------")
    print("VIEW WORKING")
    print("------------------------")
    print("Tensor after view:\n", v4)
    print("Tensor shape after view:", v4.shape)
    print("Tensor stride after view:", v4.stride())
    print("Tensor contiguity:", v4.is_contiguous(), "\n")

    v5 = v1.transpose(0, 1)
    print("------------------------")
    print("TRANSPOSE")
    print("------------------------")
    print("Tensor after transpose:", v5)
    print("Tensor contiguity:", v5.is_contiguous(), "\n")

    reshape_example(v5)

    print("------------------------")
    print("VIEW NOT WORKING")
    print("------------------------")
    try:
        v5.view(1, -1)
    except RuntimeError as e:
        print("Runtime Exception for tensor.view(1, -1):", e)


def main():
    v1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [ 9, 10, 11, 12], [13, 14, 15, 16]])

    print("------------------------")
    print("Initial tensor")
    print("------------------------")
    print(v1)
    print("Tensor shape:", v1.shape)
    print("Tensor stride:", v1.stride())
    print("Tensor contiguity:", v1.is_contiguous(), "\n")

    permute_example(v1)
    reshape_example(v1)
    view_example(v1)


if __name__ == '__main__':
    main()
