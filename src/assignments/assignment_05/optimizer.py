from .config import Config, DimType, ExecType


class Optimizer:
    def __init__(self, config: Config):
        self.config = config

    # ---------------------------------------------------------------------------
    # Task 3a – split_dim
    # ---------------------------------------------------------------------------

    def split_dim(self, dim_id: int, outer_size: int, inner_size: int):
        """Split dimension *dim_id* into an outer and an inner dimension.

        The outer dimension (left, index dim_id) gets size *outer_size*;
        the inner dimension (right, index dim_id+1) gets size *inner_size*.
        outer_size * inner_size must equal the original size.
        """
        cfg = self.config
        ndim = len(cfg.dim_sizes)

        if not (0 <= dim_id < ndim):
            raise ValueError(f"dim_id {dim_id} out of range [0, {ndim})")
        if outer_size * inner_size != cfg.dim_sizes[dim_id]:
            raise ValueError(
                f"outer_size * inner_size = {outer_size} * {inner_size} = "
                f"{outer_size * inner_size} != dim_sizes[{dim_id}] = {cfg.dim_sizes[dim_id]}"
            )

        orig_type = cfg.dim_types[dim_id]
        orig_exec = cfg.exec_types[dim_id]

        # Replace the original entry with [outer, inner] at the same position.
        for per_dim, outer_val, inner_val in [
            (cfg.dim_types,  orig_type,   orig_type),
            (cfg.exec_types, orig_exec,   orig_exec),
            (cfg.dim_sizes,  outer_size,  inner_size),
        ]:
            # remove the original dim_id entry
            per_dim.pop(dim_id)
            per_dim.insert(dim_id, inner_val)  # inserts inner at dim_id
            per_dim.insert(dim_id, outer_val)  # outer pushes inner to dim_id+1

        for t_strides in cfg.strides:
            orig_stride = t_strides.pop(dim_id)
            if orig_stride == 0:
                # Dimension was not in this tensor, so both new dims are also not in this tensor
                t_strides.insert(dim_id, 0)  # inner
                t_strides.insert(dim_id, 0)  # outer
            else:
                # Row-major: outer stride = orig * inner_size, inner = orig.
                t_strides.insert(dim_id, orig_stride)               # inner
                t_strides.insert(dim_id, orig_stride * inner_size)  # outer

    # ---------------------------------------------------------------------------
    # Task 3b – fuse_dims
    # ---------------------------------------------------------------------------

    def fuse_dims(self, dim_id_a: int, dim_id_b: int):
        """Fuse dimensions *dim_id_a* and *dim_id_b* into one.

        The two dimensions must be memory-contiguous in every tensor where
        both appear:
            stride[a] == stride[b] * size[b]   (a is outer of b)
            OR stride[b] == stride[a] * size[a] (b is outer of a)

        The fused dimension keeps the dim_type and exec_type of *dim_id_a*.
        Its size is size[a] * size[b]; its stride is the inner (smaller) one.
        """
        cfg = self.config
        ndim = len(cfg.dim_sizes)

        if not (0 <= dim_id_a < ndim):
            raise ValueError(f"dim_id_a {dim_id_a} out of range [0, {ndim})")
        if not (0 <= dim_id_b < ndim):
            raise ValueError(f"dim_id_b {dim_id_b} out of range [0, {ndim})")
        if dim_id_a == dim_id_b:
            raise ValueError("dim_id_a and dim_id_b must be different")

        size_a = cfg.dim_sizes[dim_id_a]
        size_b = cfg.dim_sizes[dim_id_b]

        # Step 1: check if the dims are contiguous in memory and have consistent relative order
        a_is_outer_dim = False
        b_is_outer_dim = False
        for tensor_id, strides in enumerate(cfg.strides):
            stride_a = strides[dim_id_a]
            stride_b = strides[dim_id_b]
            if stride_a == 0 or stride_b == 0:
                continue  # at least one dim is not in this tensor, so no contiguity requirement
            if stride_a == stride_b * size_b:
                a_is_outer_dim = True
            elif stride_b == stride_a * size_a:
                b_is_outer_dim = True
            else:
                raise ValueError(
                    f"Dimensions {dim_id_a} (stride={stride_a}, size={size_a}) and "
                    f"{dim_id_b} (stride={stride_b}, size={size_b}) are not "
                    f"memory-contiguous in tensor {tensor_id}."
                )
        if a_is_outer_dim and b_is_outer_dim:
            raise ValueError(
                f"Dimensions {dim_id_a} and {dim_id_b} have inconsistent relative order "
                f"across tensors: some tensors have {dim_id_a} as outer, others have "
                f"{dim_id_b} as outer."
            )

        # Step 2: update strides
        for strides in cfg.strides:
            stride_a = strides[dim_id_a]
            stride_b = strides[dim_id_b]
            if stride_a == 0 and stride_b == 0:
                new_stride = 0
            elif stride_a == 0:
                new_stride = stride_b
            elif stride_b == 0:
                new_stride = stride_a
            else:
                new_stride = min(stride_a, stride_b)

            strides[dim_id_a] = new_stride  # keep a
            strides.pop(dim_id_b)          # remove b

        # Step 3: update per-dimension lists
        cfg.dim_sizes[dim_id_a] = size_a * size_b
        # remove dim_id_b entries (keep dim_id_a's type and exec)
        cfg.dim_sizes.pop(dim_id_b)
        cfg.dim_types.pop(dim_id_b)
        cfg.exec_types.pop(dim_id_b)

    # -----------------------------------------------------------------------
    # Task 3c – permute_dims
    # -----------------------------------------------------------------------

    def permute_dims(self, permutation: list[int]):
        """Reorder all per-dimension lists according to *permutation*.

        Follows torch.permute semantics: the i-th entry of the result comes
        from position permutation[i] in the original.
        """
        cfg = self.config
        ndim = len(cfg.dim_sizes)

        # Check that permutation is valid
        if sorted(permutation) != list(range(ndim)):
            raise ValueError(
                f"permutation must be a permutation of [0, {ndim}), got {permutation}"
            )

        # Reorder all per-dimension lists according to the permutation
        cfg.dim_types = [cfg.dim_types[i] for i in permutation]
        cfg.exec_types = [cfg.exec_types[i] for i in permutation]
        cfg.dim_sizes = [cfg.dim_sizes[i] for i in permutation]
        cfg.strides = [[t[i] for i in permutation] for t in cfg.strides]

    # -----------------------------------------------------------------------
    # Task 3d - make_executable
    # -----------------------------------------------------------------------

    def make_executable(self):
        """Assign exec types and permute dimensions for cuTile execution.

        Strategy:
        - PRIM: one M (smallest stride in input 0), one N (smallest stride
                in input 1), one K (smallest stride in input 0).
        - PAR:  all remaining M, N, C dimensions.
        - SEQ:  all remaining K dimensions.
        Then permute to [PAR..., SEQ..., PRIM...] and call verify().
        """
        cfg = self.config

        # Step 1: reset any existing exec_types to SEQ
        cfg.exec_types = [ExecType.SEQ] * len(cfg.dim_sizes)

        def _pick_prim(dim_type: DimType, tensor_idx: int) -> int:
            """Picks the candidate with the smallest non-zero stride in tensor *tensor_idx*.
            """
            best_idx = -1
            best_stride = float("inf")
            for i, dt in enumerate(cfg.dim_types):
                if dt != dim_type:
                    # wrong dim_type, skip
                    continue
                stride = cfg.strides[tensor_idx][i]
                if 0 < stride < best_stride:
                    best_stride = stride
                    best_idx = i
            if best_idx == -1:
                raise ValueError(
                    f"No {dim_type.value} dimension found in config.")
            return best_idx

        # Step 2: assign PRIM dims based on the picking strategy
        prim_m = _pick_prim(DimType.M, 0)  # input 0 for M
        prim_n = _pick_prim(DimType.N, 1)  # input 1 for N
        prim_k = _pick_prim(DimType.K, 0)  # input 0 for K
        prim_set = {prim_m, prim_n, prim_k}

        # Step 3: assign exec types
        for i, dt in enumerate(cfg.dim_types):
            if i in prim_set:
                cfg.exec_types[i] = ExecType.PRIM
            elif dt != DimType.K:
                # M, N, C dims that are not PRIM can be PAR
                cfg.exec_types[i] = ExecType.PAR
            # else: remaining K dims stay SEQ

        # Step 4: Permute to order: PAR -> SEQ -> PRIM
        par_ids = [i for i, e in enumerate(
            cfg.exec_types) if e == ExecType.PAR]
        seq_ids = [i for i, e in enumerate(
            cfg.exec_types) if e == ExecType.SEQ]
        prim_ids = [i for i, e in enumerate(
            cfg.exec_types) if e == ExecType.PRIM]
        self.permute_dims(par_ids + seq_ids + prim_ids)

        self.verify()

    # -----------------------------------------------------------------------
    # Task 3e - verify
    # -----------------------------------------------------------------------

    def verify(self):
        """Check that the config is executable. Raises ValueError if not."""
        cfg = self.config
        exec_types = cfg.exec_types

        # 1. No K dimension may have exec_type = PAR.
        for i, (dt, et) in enumerate(zip(cfg.dim_types, exec_types)):
            if dt == DimType.K and et == ExecType.PAR:
                raise ValueError(
                    f"Dimension {i} is K but has exec_type PAR – K dims cannot be parallelised."
                )

        # Helper function: find last index of exec_type et, or -1 if not found;
        def last(et): return max(
            (i for i, e in enumerate(exec_types) if e == et), default=-1)

        # Helper function: find first index of exec_type et, or len(exec_types) if not found;
        def first(et): return next(
            (i for i, e in enumerate(exec_types) if e == et), len(exec_types))

        last_par = last(ExecType.PAR)
        first_seq = first(ExecType.SEQ)
        last_seq = last(ExecType.SEQ)
        first_prim = first(ExecType.PRIM)

        # 2. All SEQ dims must appear left of all PRIM dims.
        if last_seq > first_prim:
            raise ValueError(
                f"SEQ dimension at index {last_seq} appears after PRIM dimension "
                f"at index {first_prim}. All SEQ dims must be left of all PRIM dims."
            )

        # 3. All PAR dims must appear left of all SEQ dims.
        if last_par > first_seq:
            raise ValueError(
                f"PAR dimension at index {last_par} appears after SEQ dimension "
                f"at index {first_seq}. All PAR dims must be left of all SEQ dims."
            )

        # 4. Rightmost dims must be PRIM and include at least one M, N, K.
        # Get the indices of all PRIM dims
        prim_indices = [i for i, e in enumerate(
            exec_types) if e == ExecType.PRIM]
        if not prim_indices:
            raise ValueError("No PRIM dimensions found.")

        # Check if prim_indices are contiguous at the right end of the config
        expected = list(range(prim_indices[0], len(exec_types)))
        if prim_indices != expected:
            raise ValueError(
                f"PRIM dimensions {prim_indices} must be contiguous at the right end of the config."
            )

        # Check that the PRIM dimensions include at least one M, N, and K
        prim_dim_types = {cfg.dim_types[i] for i in prim_indices}
        for required in (DimType.M, DimType.N, DimType.K):
            if required not in prim_dim_types:
                raise ValueError(
                    f"PRIM dimensions must include at least one {required.value} dimension, "
                    f"but only found: {[dt.value for dt in prim_dim_types]}."
                )

        return True  # passed all checks

if __name__ == "__main__":
    from .config import generate_config

    print("=== split_dim ===")
    cfg = generate_config("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
    print(f"Original config:")
    print(f" dim_sizes  = {cfg.dim_sizes}")
    print(f" dim_types  = {[d.value for d in cfg.dim_types]}")
    print(f" strides[0] = {cfg.strides[0]}")
    print()
    
    opt = Optimizer(cfg)
    # Split m (dim 1, size 4096) into 16 x 256
    opt.split_dim(1, 16, 256)
    print(f"After split_dim(1, 16, 256):")
    print(f"  dim_sizes  = {cfg.dim_sizes}")
    print(f"  dim_types  = {[d.value for d in cfg.dim_types]}")
    print(f"  strides[0] = {cfg.strides[0]}")
    print()

    print("=== fuse_dims ===")
    # Fuse the two m dims back (indices 1 and 2 after split)
    opt.fuse_dims(1, 2)
    print(f"After fuse_dims(1, 2):")
    print(f"  dim_sizes  = {cfg.dim_sizes}")
    print(f"  dim_types  = {[d.value for d in cfg.dim_types]}")
    print(f"  strides[0] = {cfg.strides[0]}")
    print()

    print("=== make_executable ===")
    cfg2 = generate_config("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
    opt2 = Optimizer(cfg2)
    opt2.make_executable()
    print(cfg2)

    ###################
    # Example 2       #
    ###################
    print("=== split_dim ===")
    cfg3 = generate_config("cmk,cnk->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
    print(f"Original config:")
    print(f" dim_sizes  = {cfg3.dim_sizes}")
    print(f" dim_types  = {[d.value for d in cfg3.dim_types]}")
    print(f" strides[0] = {cfg3.strides[0]}")
    print()
    
    opt = Optimizer(cfg3)
    # Split m (dim 1, size 4096) into 16 x 256
    opt.split_dim(1, 16, 256)
    print(f"After split_dim(1, 16, 256):")
    print(f"  dim_sizes  = {cfg3.dim_sizes}")
    print(f"  dim_types  = {[d.value for d in cfg3.dim_types]}")
    print(f"  strides[0] = {cfg3.strides[0]}")
    print()

    print("=== fuse_dims ===")
    # Fuse the two m dims back (indices 1 and 2 after split)
    opt.fuse_dims(1, 2)
    print(f"After fuse_dims(1, 2):")
    print(f"  dim_sizes  = {cfg3.dim_sizes}")
    print(f"  dim_types  = {[d.value for d in cfg3.dim_types]}")
    print(f"  strides[0] = {cfg3.strides[0]}")
    print()

    print("=== make_executable ===")
    cfg4 = generate_config("cmk,cnk->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
    opt4 = Optimizer(cfg4)
    opt4.make_executable()
    print(cfg4)
