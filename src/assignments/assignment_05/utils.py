from pathlib import Path

def next_power_of_two(n: int) -> int:
	p = 1
	while p < n:
		p *= 2
	return p

def find_project_root(marker=".git") -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (no {marker} found)")

PROJECT_ROOT = find_project_root()
OUTPUT_DIR = PROJECT_ROOT / "src" / "assignments" / "assignment_05" / "resources-05"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)