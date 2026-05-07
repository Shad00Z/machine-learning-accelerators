import enum
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Task 1a – Enumeration types
# ---------------------------------------------------------------------------

class DimType(enum.Enum):
    """Classifies a loop dimension by its role in the contraction."""
    M = "M"  # appears in left input and output
    N = "N"  # appears in right input and output
    K = "K"  # appears in both inputs but not the output (contracted)
    C = "C"  # appears in all tensors (batch dimension)


class ExecType(enum.Enum):
    """Controls how a dimension is executed."""
    SEQ = "SEQ"   # sequential outer loop
    PAR = "PAR"   # parallel (mapped to block-IDs)
    PRIM = "PRIM"  # used by the primitive operation


class PrimType(enum.Enum):
    """Main (B)GEMM primitive."""
    GEMM = "GEMM"  # general matrix multiply
    BGEMM = "BGEMM"  # batched general matrix multiply


class LastType(enum.Enum):
    """Optional elementwise operation applied after accumulation."""
    NONE = "NONE"
    ELWISE_MUL = "ELWISE_MUL"


class FirstType(enum.Enum):
    """Initialisation of the accumulator."""
    ZERO = "ZERO"


class DataType(enum.Enum):
    """Numeric precision of the operands."""
    FLOAT16 = "FLOAT16"
    FLOAT32 = "FLOAT32"


# ---------------------------------------------------------------------------
# Task 1b – Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Represents a tensor contraction configuration.

    Attributes
    ----------
    data_type:
        Numeric precision of the operands.
    prim_main:
        Main primitive used inside the kernel.
    prim_last:
        Optional elementwise operation applied after accumulation.
    prim_first:
        Initialisation of the accumulator.
    dim_types:
        Per-dimension index type (one entry per dimension).
    exec_types:
        Per-dimension execution type (one entry per dimension).
    dim_sizes:
        Per-dimension size (one entry per dimension).
    strides:
        Per-tensor, per-dimension stride list.
        ``strides[t][d]`` is the stride of tensor *t* along dimension *d*.
        A value of 0 means the dimension does not appear in that tensor.
    """

    data_type:  DataType
    prim_main:  PrimType
    prim_last:  LastType
    prim_first: FirstType

    dim_types:  list[DimType] = field(default_factory=list)
    exec_types: list[ExecType] = field(default_factory=list)
    dim_sizes:  list[int] = field(default_factory=list)
    # strides[tensor_idx][dim_idx]
    strides:    list[list[int]] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [
            "Config(",
            f"  data_type  = {self.data_type}",
            f"  prim_main  = {self.prim_main}",
            f"  prim_last  = {self.prim_last}",
            f"  prim_first = {self.prim_first}",
            f"  dim_types  = {[d.value for d in self.dim_types]}",
            f"  exec_types = {[e.value for e in self.exec_types]}",
            f"  dim_sizes  = {self.dim_sizes}",
        ]
        for i, s in enumerate(self.strides):
            lines.append(f"  strides[{i}] = {s}")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 2 – generate_config
# ---------------------------------------------------------------------------

def _parse_einsum(einsum_str: str) -> tuple[list[str], list[list[str]]]:
    """Return (output_indices, list_of_input_indices) from an einsum string.

    Supports the ``'ij,jk->ik'`` notation using any single-character
    indices.
    """
    einsum_str = einsum_str.replace(" ", "")
    if "->" not in einsum_str:
        raise ValueError(f"Einsum string must contain '->': {einsum_str!r}")
    inputs_str, output_str = einsum_str.split("->")
    input_parts = inputs_str.split(",")
    return list(output_str), [list(p) for p in input_parts]


def _classify_dim(idx: str,
                  output_indices: list[str],
                  input_indices: list[list[str]]) -> DimType:
    """Classify a single dimension index.

    Rules:
    - M: Appears in output and left input
    - N: Appears in output and right input
    - C: Appears in all tensors (both inputs + output)
    - K: Appears in both inputs but not output
    """
    in_output = idx in output_indices
    in_left = len(input_indices) > 0 and idx in input_indices[0]
    in_right = len(input_indices) > 1 and idx in input_indices[1]

    if in_output and in_left and in_right:
        return DimType.C
    if in_output and in_left:
        return DimType.M
    if in_output and in_right:
        return DimType.N
    if in_left and in_right:
        return DimType.K
    raise ValueError(
        f"Cannot classify dimension '{idx}' in einsum expression.")


def _compute_row_major_strides(indices: list[str],
                               dim_order: list[str],
                               dim_sizes: dict[str, int]) -> list[int]:
    """Compute row-major strides for a tensor whose axes are *indices*.

    *dim_order* is the order of all dimensions in the Config.
    Returns a list of length ``len(dim_order)`` where dimensions not in
    *indices* get a stride of 0.
    """
    # rightmost index gets stride 1
    # then multiply by size as we move leftwards
    tensor_strides: dict[str, int] = {}
    stride = 1
    for idx in reversed(indices):
        tensor_strides[idx] = stride
        stride *= dim_sizes[idx]

    # 0 as default
    return [tensor_strides.get(d, 0) for d in dim_order]


def generate_config(einsum_str: str, input_shapes: list[tuple[int, ...]]) -> Config:
    """Generate a :class:`Config` from an einsum string and input shapes.

    Parameters
    ----------
    einsum_str:
        Einsum notation, e.g. ``'cmk,ckn->cmn'``.
    input_shapes:
        A list of shape tuples, one per input tensor.
        The output shape is implied by the einsum.

    Returns
    -------
    Config
        A fully initialised Config with all ``exec_types`` set to ``SEQ``,
        ``data_type = FLOAT16``, ``prim_main = GEMM``, ``prim_last = NONE``,
        ``prim_first = ZERO``.
    """
    # Step 1: Parse the einsum string to get output and input indices
    output_indices, input_indices_list = _parse_einsum(einsum_str)

    if len(input_indices_list) != len(input_shapes):
        raise ValueError(
            f"Number of input tensors in einsum ({len(input_indices_list)}) "
            f"does not match number of provided shapes ({len(input_shapes)})."
        )

    # Step 2: Collect dimension sizes and check for consistency
    dim_sizes: dict[str, int] = {}
    # For each input tensor (list of indices) and its shape
    for indices, shape in zip(input_indices_list, input_shapes):
        # rank mismatch check
        if len(indices) != len(shape):
            raise ValueError(
                f"Input tensor with indices {''.join(indices)!r} has "
                f"{len(indices)} indices but shape {shape} has rank {len(shape)}."
            )
        # For each input dimension index and its corresponding size
        for idx, sz in zip(indices, shape):
            if idx in dim_sizes and dim_sizes[idx] != sz:
                # We already have a size for this dimension index, but it conflicts with the new one
                raise ValueError(
                    f"Conflicting sizes for dimension '{idx}': "
                    f"{dim_sizes[idx]} vs {sz}."
                )
            dim_sizes[idx] = sz

    # Step 3: Build per-dimension attribute lists, one entry per key in dim_sizes
    dim_types_list:  list[DimType]  = []
    exec_types_list: list[ExecType] = []
    sizes_list:      list[int]      = []
    for idx, sz in dim_sizes.items():
        dim_types_list.append(_classify_dim(idx, output_indices, input_indices_list))
        exec_types_list.append(ExecType.SEQ)
        sizes_list.append(sz)

    # Step 4: Compute strides for each tensor
    all_tensor_indices: list[list[str]] = input_indices_list + [output_indices]
    strides_list: list[list[int]] = [
        _compute_row_major_strides(ti, list(dim_sizes.keys()), dim_sizes)
        for ti in all_tensor_indices
    ]

    return Config(
        data_type=DataType.FLOAT16,
        prim_main=PrimType.GEMM,
        prim_last=LastType.NONE,
        prim_first=FirstType.ZERO,
        dim_types=dim_types_list,
        exec_types=exec_types_list,
        dim_sizes=sizes_list,
        strides=strides_list,
    )


if __name__ == "__main__":
    cfg = generate_config("ij,jk->ik", [(4, 8), (8, 16)])
    print("=== ij,jk->ik ===")
    print(cfg)

    print()

    cfg2 = generate_config("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
    print("=== cmk,ckn->cmn ===")
    print(cfg2)
