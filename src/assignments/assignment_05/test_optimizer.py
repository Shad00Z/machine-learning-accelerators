import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from .config import Config, DimType, ExecType, generate_config
from .optimizer import Optimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_opt(einsum: str, shapes: list[tuple[int, ...]]) -> tuple[Config, Optimizer]:
    cfg = generate_config(einsum, shapes)
    return cfg, Optimizer(cfg)


# ---------------------------------------------------------------------------
# split_dim
# ---------------------------------------------------------------------------

class TestSplitDim:
    def test_basic_split(self):
        cfg, opt = make_opt("mk,kn->mn", [(4096, 4096), (4096, 4096)])
        # dim 0 is M (size 4096), split into 16 x 256
        opt.split_dim(0, 16, 256)
        assert cfg.dim_sizes[0] == 16
        assert cfg.dim_sizes[1] == 256
        assert cfg.dim_types[0] == DimType.M
        assert cfg.dim_types[1] == DimType.M

    def test_stride_update_inner_dim(self):
        # cmk,ckn->cmn: dims are c(0), m(1), k(2)
        cfg, opt = make_opt("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
        # strides[0] before: c=4096*4096, m=4096, k=1
        m_stride_before = cfg.strides[0][1]  # stride of m in tensor 0
        opt.split_dim(1, 16, 256)
        # outer (dim 1) should have stride = m_stride_before * 256
        # inner (dim 2) should have stride = m_stride_before
        assert cfg.strides[0][1] == m_stride_before * 256
        assert cfg.strides[0][2] == m_stride_before

    def test_zero_stride_preserved(self):
        # k does not appear in output tensor (stride 0 there)
        cfg, opt = make_opt("mk,kn->mn", [(4096, 4096), (4096, 4096)])
        k_idx = next(i for i, d in enumerate(cfg.dim_types) if d == DimType.K)
        opt.split_dim(k_idx, 64, 64)
        # output is tensor index 2; both new k dims should be 0 there
        assert cfg.strides[2][k_idx] == 0
        assert cfg.strides[2][k_idx + 1] == 0

    def test_invalid_dim_id(self):
        _, opt = make_opt("mk,kn->mn", [(4, 4), (4, 4)])
        with pytest.raises(ValueError, match="out of range"):
            opt.split_dim(10, 2, 2)

    def test_size_mismatch(self):
        _, opt = make_opt("mk,kn->mn", [(4096, 4096), (4096, 4096)])
        with pytest.raises(ValueError, match=r"outer_size \* inner_size"):
            opt.split_dim(0, 100, 100)  # 100*100 != 4096


# ---------------------------------------------------------------------------
# fuse_dims
# ---------------------------------------------------------------------------

class TestFuseDims:
    def test_fuse_roundtrip(self):
        """split then fuse should restore the original state."""
        cfg, opt = make_opt("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
        original_sizes = list(cfg.dim_sizes)
        original_strides = [list(s) for s in cfg.strides]

        opt.split_dim(1, 16, 256)
        opt.fuse_dims(1, 2)

        assert cfg.dim_sizes == original_sizes
        assert cfg.strides == original_strides

    def test_fused_size(self):
        cfg, opt = make_opt("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
        opt.split_dim(1, 16, 256)
        # After split: dims are c(0), m_outer(1,sz=16), m_inner(2,sz=256), k(3)
        opt.fuse_dims(1, 2)
        assert cfg.dim_sizes[1] == 16 * 256

    def test_fused_stride_is_inner(self):
        cfg, opt = make_opt("mk,kn->mn", [(4096, 4096), (4096, 4096)])
        opt.split_dim(0, 64, 64)          # m_outer(0, stride=64), m_inner(1, stride=1)
        stride_inner_before = cfg.strides[0][1]
        opt.fuse_dims(0, 1)
        assert cfg.strides[0][0] == stride_inner_before

    def test_inconsistent_order_raises(self):
        """Dims that are outer in one tensor but inner in another must be rejected."""
        # Build a config manually where a is outer in tensor 0 but b is outer in tensor 1
        cfg = generate_config("mk,kn->mn", [(4, 4), (4, 4)])
        # Manually patch strides so dim 0 and dim 1 are contiguous but with flipped order
        # dim_sizes: [4, 4, 4]  (m=4, k=4, n=4)
        # tensor 0 (mk): stride m=4, k=1  -> m is outer of k  (stride_m == stride_k * size_k)
        # tensor 1 (kn): stride k=4, n=1  -> k is outer of n
        # tensor 2 (mn): stride m=4, n=1  -> m is outer of n
        # dims: m(0), k(1), n(2)
        # Let's manually make tensor 0 say k is outer of m and tensor 1 say m is outer of k:
        # size_m=4, size_k=4
        # tensor 0: stride_k=4, stride_m=1  -> k outer of m
        # tensor 1: stride_m=4, stride_k=1  -> m outer of k
        cfg.strides[0] = [1, 4, 0]   # stride_m=1, stride_k=4 => k is outer of m
        cfg.strides[1] = [4, 1, 0]   # stride_m=4, stride_k=1 => m is outer of k (flipped)
        # tensor 2 (output mn) doesn't contain k, so it won't affect the order check
        opt = Optimizer(cfg)
        with pytest.raises(ValueError, match="inconsistent relative order"):
            opt.fuse_dims(0, 1)  # fuse m and k

    def test_same_dim_raises(self):
        _, opt = make_opt("mk,kn->mn", [(4, 4), (4, 4)])
        with pytest.raises(ValueError, match="must be different"):
            opt.fuse_dims(0, 0)

    def test_non_contiguous_raises(self):
        cfg, opt = make_opt("cmk,ckn->cmn", [(4, 16, 32), (4, 32, 16)])
        # c(0) and n(3) are not contiguous
        with pytest.raises(ValueError, match="not memory-contiguous"):
            opt.fuse_dims(0, 3)


# ---------------------------------------------------------------------------
# permute_dims
# ---------------------------------------------------------------------------

class TestPermuteDims:
    def test_identity_permutation(self):
        cfg, opt = make_opt("mk,kn->mn", [(4, 4), (4, 4)])
        sizes_before = list(cfg.dim_sizes)
        types_before = list(cfg.dim_types)
        opt.permute_dims([0, 1, 2])
        assert cfg.dim_sizes == sizes_before
        assert cfg.dim_types == types_before

    def test_reverse_permutation(self):
        cfg, opt = make_opt("mk,kn->mn", [(4, 8), (8, 16)])
        # dims: m(sz=4), k(sz=8), n(sz=16)
        sizes_before = list(cfg.dim_sizes)
        opt.permute_dims([2, 1, 0])
        assert cfg.dim_sizes == list(reversed(sizes_before))

    def test_strides_permuted_correctly(self):
        cfg, opt = make_opt("mk,kn->mn", [(4, 8), (8, 16)])
        strides_before = [list(s) for s in cfg.strides]
        opt.permute_dims([2, 1, 0])
        for t, (before, after) in enumerate(zip(strides_before, cfg.strides)):
            assert after == [before[2], before[1], before[0]]

    def test_invalid_permutation(self):
        _, opt = make_opt("mk,kn->mn", [(4, 4), (4, 4)])
        with pytest.raises(ValueError, match="permutation"):
            opt.permute_dims([0, 0, 1])


# ---------------------------------------------------------------------------
# make_executable
# ---------------------------------------------------------------------------

class TestMakeExecutable:
    def test_produces_valid_config(self):
        _, opt = make_opt("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
        opt.make_executable()
        assert opt.verify() is True

    def test_prim_order_par_seq_prim(self):
        cfg, opt = make_opt("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
        opt.make_executable()
        exec_types = cfg.exec_types
        # PAR dims come first, then SEQ, then PRIM
        seen_seq = seen_prim = False
        for et in exec_types:
            if et == ExecType.SEQ:
                seen_seq = True
            if et == ExecType.PRIM:
                seen_prim = True
            if et == ExecType.PAR:
                assert not seen_seq and not seen_prim, "PAR appears after SEQ or PRIM"
            if et == ExecType.SEQ:
                assert not seen_prim, "SEQ appears after PRIM"

    def test_prim_contains_m_n_k(self):
        cfg, opt = make_opt("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
        opt.make_executable()
        prim_types = {cfg.dim_types[i] for i, e in enumerate(cfg.exec_types) if e == ExecType.PRIM}
        assert DimType.M in prim_types
        assert DimType.N in prim_types
        assert DimType.K in prim_types

    def test_no_k_dim_is_par(self):
        cfg, opt = make_opt("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
        opt.make_executable()
        for dt, et in zip(cfg.dim_types, cfg.exec_types):
            assert not (dt == DimType.K and et == ExecType.PAR)

    def test_transposed_input(self):
        """make_executable should also work for transposed layouts."""
        _, opt = make_opt("cmk,cnk->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
        opt.make_executable()
        assert opt.verify() is True


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

class TestVerify:
    def test_valid_config_passes(self):
        _, opt = make_opt("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)])
        opt.make_executable()
        assert opt.verify() is True

    def test_k_as_par_raises(self):
        cfg, opt = make_opt("mk,kn->mn", [(4, 4), (4, 4)])
        opt.make_executable()
        # Force a K dim to PAR
        k_idx = next(i for i, d in enumerate(cfg.dim_types) if d == DimType.K)
        cfg.exec_types[k_idx] = ExecType.PAR
        with pytest.raises(ValueError, match="K but has exec_type PAR"):
            opt.verify()

    def test_seq_after_prim_raises(self):
        cfg, opt = make_opt("mk,kn->mn", [(4, 4), (4, 4)])
        opt.make_executable()
        # Swap a SEQ and PRIM entry to violate ordering
        prim_idx = next(i for i, e in enumerate(cfg.exec_types) if e == ExecType.PRIM)
        # Place a SEQ after PRIM
        cfg.exec_types.append(ExecType.SEQ)
        cfg.dim_types.append(DimType.K)
        cfg.dim_sizes.append(1)
        for s in cfg.strides:
            s.append(0)
        with pytest.raises(ValueError, match="SEQ dimension"):
            opt.verify()

    def test_no_prim_raises(self):
        cfg, opt = make_opt("mk,kn->mn", [(4, 4), (4, 4)])
        # Set all exec types to SEQ
        cfg.exec_types = [ExecType.SEQ] * len(cfg.exec_types)
        with pytest.raises(ValueError, match="No PRIM dimensions found"):
            opt.verify()
