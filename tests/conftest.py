import mlx.core as mx
import pytest


@pytest.fixture(
    params=[
        pytest.param((64, 256), id="tiny"),
        pytest.param((4096, 4096), id="large"),
        pytest.param((16, 16, 64, 256), id="4d"),
    ]
)
def weight(request):
    mx.random.seed(0)
    return mx.random.normal(request.param)
