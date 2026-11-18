import mlx.core as mx
import pytest

from quant_lab.quant.pack import pack_codes, unpack_codes


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("shape", [(64, 128), (16, 4096, 4096)])
def test_pack_unpack_roundtrip(bits, shape):
    weight = mx.random.randint(0, 1 << bits, shape).astype(mx.uint8)
    out = unpack_codes(pack_codes(weight, bits), bits, weight.dtype)
    assert weight.shape == out.shape
    assert mx.array_equal(weight, out).item()
