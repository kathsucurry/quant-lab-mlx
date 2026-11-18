import mlx.core as mx

from .config import QuantProperties

EPS = 1e-7


def affine_quantize(
    weight: mx.array,
    quant_props: QuantProperties,
) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize weight given the quantization properties.

    Returns:
        A tuple of (packed weight as uint8, scale, biases).
        Scale and biases have the same dtype as weight.
    """
    n_bins = (1 << quant_props.bits) - 1
    if weight.shape[-1] % quant_props.group_size != 0:
        raise ValueError(
            "Quantization: the weight's last dimension is not divisible by group size."
        )

    # Ensure the quantization runs in fp32 for better precision.
    group_weight = weight.astype(mx.float32).reshape(*weight.shape[:-1], -1, quant_props.group_size)

    # Obtain min and max values.
    group_weight_min = group_weight.min(axis=-1)
    group_weight_max = group_weight.max(axis=-1)

    group_scales = mx.maximum((group_weight_max - group_weight_min) / n_bins, EPS)
    group_bias = group_weight_min

    group_quantized = mx.clip(
        mx.round((group_weight - group_bias[..., None]) / group_scales[..., None]), 0, n_bins
    )

    return (group_quantized.reshape(weight.shape), group_scales, group_bias)


def affine_dequantize():
    pass
