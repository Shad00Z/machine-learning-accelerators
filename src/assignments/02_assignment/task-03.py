import torch
import cuda.tile as ct
import triton

ConstInt = ct.Constant[int]

# ===========================================================================
# Task 3: 4D Tensor Element-wise Addition
# ===========================================================================

def next_power_of_two(n: int) -> int:
	p = 1
	while p < n:
		p *= 2
	return p


@ct.kernel
def tensor_elementwise_addition_kl_kernel(A, B, C, tK: ConstInt, tL: ConstInt):
	# Tiling over (K, L) and parallelization over (M, N).
	m = ct.bid(0)
	n = ct.bid(1)

	tile_a = ct.load(
		A,
		index=(m, n, 0, 0),
		shape=(1, 1, tK, tL),
		padding_mode=ct.PaddingMode.ZERO,
	)
	tile_b = ct.load(
		B,
		index=(m, n, 0, 0),
		shape=(1, 1, tK, tL),
		padding_mode=ct.PaddingMode.ZERO,
	)

	tile_c = tile_a + tile_b
	ct.store(C, index=(m, n, 0, 0), tile=tile_c)


@ct.kernel
def tensor_elementwise_addition_mn_kernel(A, B, C, tM: ConstInt, tN: ConstInt):
	# Tiling over (M, N) and parallelization over (K, L).
	k = ct.bid(0)
	l = ct.bid(1)

	tile_a = ct.load(
		A,
		index=(0, 0, k, l),
		shape=(tM, tN, 1, 1),
		padding_mode=ct.PaddingMode.ZERO,
	)
	tile_b = ct.load(
		B,
		index=(0, 0, k, l),
		shape=(tM, tN, 1, 1),
		padding_mode=ct.PaddingMode.ZERO,
	)

	tile_c = tile_a + tile_b
	ct.store(C, index=(0, 0, k, l), tile=tile_c)


def tensor_elementwise_addition_kl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
	if A.shape != B.shape:
		raise ValueError("Input tensors must have the same shape.")
	if A.device != B.device:
		raise ValueError("Input tensors must be on the same device.")
	if not A.is_cuda or not B.is_cuda:
		raise ValueError("Input tensors must be on CUDA.")

	M, N, K, L = A.shape
	tK = next_power_of_two(K)
	tL = next_power_of_two(L)

	grid = (M, N, 1)
	C = torch.empty_like(A)

	ct.launch(
		torch.cuda.current_stream(),
		grid,
		tensor_elementwise_addition_kl_kernel,
		(A, B, C, tK, tL),
	)
	return C


def tensor_elementwise_addition_mn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
	if A.shape != B.shape:
		raise ValueError("Input tensors must have the same shape.")
	if A.device != B.device:
		raise ValueError("Input tensors must be on the same device.")
	if not A.is_cuda or not B.is_cuda:
		raise ValueError("Input tensors must be on CUDA.")

	M, N, K, L = A.shape
	tM = next_power_of_two(M)
	tN = next_power_of_two(N)

	grid = (K, L, 1)
	C = torch.empty_like(A)

	ct.launch(
		torch.cuda.current_stream(),
		grid,
		tensor_elementwise_addition_mn_kernel,
		(A, B, C, tM, tN),
	)
	return C


def test_tensor_elementwise_addition_kl():
	M, N, K, L = 21, 99, 9, 76
	A = torch.rand((M, N, K, L), dtype=torch.float16, device="cuda")
	B = torch.rand((M, N, K, L), dtype=torch.float16, device="cuda")

	result = tensor_elementwise_addition_kl(A, B)
	expected = A + B
	assert torch.allclose(result, expected, rtol=1e-2), "KL-tile 4D addition failed"


def test_tensor_elementwise_addition_mn():
	M, N, K, L = 21, 99, 9, 76
	A = torch.rand((M, N, K, L), dtype=torch.float16, device="cuda")
	B = torch.rand((M, N, K, L), dtype=torch.float16, device="cuda")

	result = tensor_elementwise_addition_mn(A, B)
	expected = A + B
	assert torch.allclose(result, expected, rtol=1e-2), "MN-tile 4D addition failed"


def run_benchmark(warmup: int = 200, iters: int = 2000):
	M, N, K, L = 16, 128, 16, 128
	A = torch.rand((M, N, K, L), dtype=torch.float16, device="cuda")
	B = torch.rand((M, N, K, L), dtype=torch.float16, device="cuda")
	stream = torch.cuda.current_stream()

	# Precompute launch parameters
	tK = next_power_of_two(K)
	tL = next_power_of_two(L)
	tM = next_power_of_two(M)
	tN = next_power_of_two(N)
	grid_kl = (M, N, 1)
	grid_mn = (K, L, 1)

	C_kl = torch.empty_like(A)
	C_mn = torch.empty_like(A)
	C_torch = torch.empty_like(A)

	def launch_kl():
		ct.launch(
			stream,
			grid_kl,
			tensor_elementwise_addition_kl_kernel,
			(A, B, C_kl, tK, tL),
		)

	def launch_mn():
		ct.launch(
			stream,
			grid_mn,
			tensor_elementwise_addition_mn_kernel,
			(A, B, C_mn, tM, tN),
		)

	def launch_torch():
		torch.add(A, B, out=C_torch)

	kl_ms = triton.testing.do_bench(launch_kl, warmup=warmup, rep=iters)
	mn_ms = triton.testing.do_bench(launch_mn, warmup=warmup, rep=iters)
	torch_ms = triton.testing.do_bench(launch_torch, warmup=warmup, rep=iters)

	print("Benchmark (ms per launch):")
	print(f"  cuTile KL-tiling: {kl_ms:.4f} ms")
	print(f"  cuTile MN-tiling: {mn_ms:.4f} ms")
	print(f"  torch.add:  {torch_ms:.4f} ms")


if __name__ == "__main__":
	test_tensor_elementwise_addition_kl()
	test_tensor_elementwise_addition_mn()
	print("Task 3 tests passed.")
	run_benchmark()
