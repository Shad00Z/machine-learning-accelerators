import random
import torch


def next_power_of_two(n: int) -> int:
	p = 1
	while p < n:
		p *= 2
	return p


def tensor_initialization() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Initialize random tensors
    a = random.randint(2, 32)
    b = random.randint(2, 32)
    c = random.randint(2, 32)
    e = random.randint(2, 32)
    k = random.randint(2, 32)
    l = random.randint(2, 32)
    x = random.randint(2, 32)
    y = random.randint(2, 32)
    z = random.randint(2, 32)
    
    size_A = e * a * b * k * l * x * y
    size_B = e * c * k * l * y * z
    size_C = e * a * b * c * x * z
    
    while (size_A + size_B) * 2 + size_C * 4 > 34 * 1024**3:
        a = random.randint(2, a)
        b = random.randint(2, b)
        c = random.randint(2, c)
        e = random.randint(2, e)
        k = random.randint(2, k)
        l = random.randint(2, l)
        x = random.randint(2, x)
        y = random.randint(2, y)
        z = random.randint(2, z)
        
        size_A = e * a * b * k * l * x * y
        size_B = e * c * k * l * y * z
        size_C = e * a * b * c * x * z
    
    A = torch.rand((e, a, b, k, l, x, y), dtype=torch.float16, device="cuda")
    B = torch.rand((e, c, k, l, y, z), dtype=torch.float16, device="cuda")
    C = torch.zeros((e, a, b, c , x, z), dtype=torch.float32, device="cuda")
    
    return [A, B, C]
