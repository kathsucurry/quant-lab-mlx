import pytest

from quant_lab.quant.config import OPTIONS_GROUP_SIZE, QuantProperties


@pytest.mark.parametrize("group_size", [0, 3, 64, -64, 8, 24, 10, -16, 32.0, 64.0])
def test_rejects_invalid_group_size(group_size):
    if group_size not in OPTIONS_GROUP_SIZE or not isinstance(group_size, int):
        with pytest.raises(ValueError):
            QuantProperties(group_size=group_size)


@pytest.mark.parametrize("mode", ["", None])
def test_rejects_invalid_mode(mode):
    with pytest.raises(ValueError):
        QuantProperties(mode=mode)
