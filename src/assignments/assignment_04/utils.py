import cuda.tile as ct
import random
import torch

from pathlib import Path

def next_power_of_two(n: int) -> int:
	p = 1
	while p < n:
		p *= 2
	return p


def tensor_initialization() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Initialize random tensors
    a = random.randint(4, 32)
    b = random.randint(4, 32)
    c = random.randint(4, 32)
    e = random.randint(4, 32)
    k = random.randint(4, 32)
    l = random.randint(4, 32)
    x = random.randint(4, 32)
    y = random.randint(4, 32)
    z = random.randint(4, 32)
    
    size_A = e * a * b * k * l * x * y
    size_B = e * c * k * l * y * z
    size_C = e * a * b * c * x * z
    
    while (size_A + size_B) * 2 + size_C * 4 > 34 * 1024**3:
        a = random.randint(4, a)
        b = random.randint(4, b)
        c = random.randint(4, c)
        e = random.randint(4, e)
        k = random.randint(4, k)
        l = random.randint(4, l)
        x = random.randint(4, x)
        y = random.randint(4, y)
        z = random.randint(4, z)
        
        size_A = e * a * b * k * l * x * y
        size_B = e * c * k * l * y * z
        size_C = e * a * b * c * x * z
    
    A = torch.rand((e, a, b, k, l, x, y), dtype=torch.float16, device="cuda")
    B = torch.rand((e, c, k, l, y, z), dtype=torch.float16, device="cuda")
    C = torch.zeros((e, a, b, c , x, z), dtype=torch.float32, device="cuda")
    
    return [A, B, C]

def find_project_root(marker=".git") -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (no {marker} found)")

PROJECT_ROOT = find_project_root()
OUTPUT_DIR = PROJECT_ROOT / "src" / "assignments" / "assignment_04"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
