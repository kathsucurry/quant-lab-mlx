import math

import mlx.core as mx


def _get_pack_dtype(bits: int) -> mx.Dtype:
    if bits == 5:
        return mx.uint64
    if bits in [3, 6]:
        return mx.uint32
    return mx.uint8


def pack_codes(weight: mx.array, bits: int) -> mx.array:
    # Compute the number of quantized values in each pack.
    n_vals_in_pack = math.lcm(bits, 8) // bits
    bytes_per_pack = n_vals_in_pack * bits // 8
    pack_dtype = _get_pack_dtype(bits)
    shifts = mx.arange(n_vals_in_pack, dtype=pack_dtype) * bits
    bytes_shifts = mx.arange(bytes_per_pack, dtype=pack_dtype) * 8

    weight = weight.reshape(*weight.shape[:-1], -1, n_vals_in_pack).astype(pack_dtype)
    weight = (weight << shifts).sum(axis=-1, keepdims=True)
    weight = ((weight >> bytes_shifts) & 0xFF).astype(mx.uint8).reshape(*weight.shape[:-2], -1)

    return weight


def unpack_codes(weight: mx.array, bits: int, out_dtype: mx.Dtype) -> mx.array:
    n_vals_in_pack = math.lcm(bits, 8) // bits
    bytes_per_pack = n_vals_in_pack * bits // 8
    pack_dtype = _get_pack_dtype(bits)
    shifts = mx.arange(n_vals_in_pack, dtype=pack_dtype) * bits
    bytes_shifts = mx.arange(bytes_per_pack, dtype=pack_dtype) * 8

    weight = (
        weight.reshape(*weight.shape[:-1], -1, bytes_per_pack).astype(pack_dtype) << bytes_shifts
    ).sum(axis=-1, keepdims=True)
    weight = (weight >> shifts) & ((1 << bits) - 1)

    return weight.reshape(*weight.shape[:-2], -1).astype(out_dtype)
