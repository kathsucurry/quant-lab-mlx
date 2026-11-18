import mlx.core as mx

from .config import QuantProperties

EPS = 1e-7


def affine_quantize(
    weight: mx.array,
    quant_props: QuantProperties,
) -> tuple[mx.array, mx.array, mx.array]:
    """Quantizes weight given the quantization properties.

    TODO:
        - packing
        - currently, the implementation always produces scales and bias with the same dtype
            as weight.

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
        mx.round((group_weight - group_bias[..., None]) / group_scales[..., None]).astype(mx.uint8),
        0,
        n_bins,
    )

    return (group_quantized.reshape(weight.shape), group_scales, group_bias)


def affine_dequantize(
    quantized_weight: mx.array,
    scales: mx.array,
    bias: mx.array,
    quant_props: QuantProperties,
) -> mx.array:
    """Dequantizes quantized weight given the quantization properties, scales, and bias.

    TODO:
        - handle packing
        - currently, the implementation assumes that the dequantized weight would have
            the same dtype as scales/bias.

    Returns:
        The dequantized weight.
    """
    out_dtype = scales.dtype

    weight = quantized_weight.reshape(
        *quantized_weight.shape[:-1], -1, quant_props.group_size
    ).astype(out_dtype)
    weight = weight * scales[..., None] + bias[..., None]

    return weight.reshape(quantized_weight.shape)
