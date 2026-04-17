import torch
import cuda.tile as ct

# ===========================================================================
# Task 2: Matrix Reduction Kernel
# ===========================================================================

@ct.kernel
def matrix_sum_reduction_kernel(input_matrix, output_vector, tM: ct.Constant[int], tK: ct.Constant[int]):
    block_id = ct.bid(0)
    # load a tile of shape (tM, tK) with zero-padding
    # -> only one tile in K dimension (index 0)
    input_tile = ct.load(input_matrix, index=(block_id, 0), shape=(tM, tK),
                         padding_mode=ct.PaddingMode.ZERO).astype(input_matrix.dtype)
    # sum along the columns to get a tile of shape (tM,)
    output_tile = ct.sum(input_tile, axis=1)
    ct.store(output_vector, index=(block_id,), tile=output_tile)
    
def matrix_sum_reduction(input_matrix, tM):
    num_rows = input_matrix.shape[0]
    num_cols = input_matrix.shape[1]
    # tile dimensions must be powers of 2 -> round up
    tK = 1
    while tK < num_cols:
        tK *= 2
    
    grid = (ct.cdiv(num_rows, tM), 1, 1)
    output_vector = torch.zeros(num_rows, dtype=input_matrix.dtype, device='cuda')

    ct.launch(torch.cuda.current_stream(),
              grid,
              matrix_sum_reduction_kernel,
              (input_matrix, output_vector, tM, tK))
    
    return output_vector

def test_matrix_sum_reduction():
    M = 2**7
    K = 2**5
    tM = 2**4

    input_matrix = torch.rand((M, K), dtype=torch.float16, device='cuda')
    output_vector = matrix_sum_reduction(input_matrix, tM)
    
    expected_output = torch.sum(input_matrix, dim=1)
    assert torch.allclose(output_vector, expected_output, rtol=1e-2), "Matrix sum reduction failed!"

if __name__ == "__main__":
    test_matrix_sum_reduction()