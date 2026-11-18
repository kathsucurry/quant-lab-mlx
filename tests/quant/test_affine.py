import mlx.core as mx
import pytest

from quant_lab.quant.affine import affine_dequantize, affine_quantize
from quant_lab.quant.config import QuantProperties


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_roundtrip_error_within_half_scale(weight, bits, group_size):
    quant_props = QuantProperties(group_size, bits, "affine")
    quant, scales, bias = affine_quantize(weight, quant_props)
    weight_hat = affine_dequantize(quant, scales, bias, quant_props)

    err = mx.abs(weight - weight_hat).reshape(*weight.shape[:-1], -1, group_size).max(axis=-1)
    assert mx.all(err <= scales / 2 + 1e-6).item()


@pytest.mark.parametrize("constant_value", [0.0, -1.0, 1.0, -2.0, 2.0])
def test_constant_group_values_roundtrips(constant_value):
    quant_props = QuantProperties(group_size=64, bits=4)
    weight = mx.full((4, 128), constant_value)

    quant, scales, bias = affine_quantize(weight, quant_props)
    weight_hat = affine_dequantize(quant, scales, bias, quant_props)

    assert mx.all(scales > 0).item()
    assert not mx.any(mx.isnan(weight_hat)).item()
    assert mx.allclose(weight_hat, weight).item()
