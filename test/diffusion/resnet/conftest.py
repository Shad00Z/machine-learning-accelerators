import pytest

from data.load_helper import data_path, load_data


class TensorBatch(dict):
    """Dict subclass that moves all tensor values to CUDA on .cuda()."""

    def cuda(self):
        for k, v in self.items():
            if hasattr(v, "cuda"):
                self[k] = v.cuda()
        return self


@pytest.fixture(scope="module")
def ref():
    path = data_path()
    if not path.exists():
        pytest.skip(f"reference data not found: {path}")
    return TensorBatch(load_data(path))
