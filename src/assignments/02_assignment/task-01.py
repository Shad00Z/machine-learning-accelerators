import cupy as cp

# ===========================================================================
# Task 1: GPU Device Properties
# ===========================================================================

cuda_attributes = cp.cuda.Device().attributes

l2_cache_size = cuda_attributes['L2CacheSize']
max_smem_per_sm = cuda_attributes['MaxSharedMemoryPerMultiprocessor']
clock_rate_khz = cuda_attributes['ClockRate']

print("Task 1: GPU Device Properties")
print(f"L2CacheSize: {l2_cache_size / (1024 ** 2):.2f} MiB")
print(f"MaxSharedMemoryPerMultiprocessor: {max_smem_per_sm / 1024:.1f} KiB")
print(f"ClockRate: {clock_rate_khz / 1_000_000:.3f} GHz")