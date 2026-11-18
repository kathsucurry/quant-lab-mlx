OPTIONS_BITS = [2, 3, 4, 5, 6, 8]
OPTIONS_GROUP_SIZE = [16, 32, 64, 128]
OPTIONS_MODE = ["affine"]


class QuantProperties:
    """Quantization requires group size, bits, and mode."""

    def __init__(self, group_size: int = 64, bits: int = 8, mode: str = "affine"):
        self.group_size: int = int(group_size)
        self.bits: int = int(bits)
        self.mode: str = mode
        self._validate_properties()

    def _validate_properties(self):
        if self.bits not in OPTIONS_BITS:
            raise ValueError(
                f"Quantizing to {self.bits} bits is not available. "
                + f"Possible options: {OPTIONS_BITS}"
            )

        if self.mode not in OPTIONS_MODE:
            raise ValueError(
                f"Quantizing with mode {self.mode} is not available. "
                + f"Possible options: {OPTIONS_MODE}"
            )

        if self.group_size not in OPTIONS_GROUP_SIZE:
            raise ValueError(
                f"Quantizing with group size {self.group_size} is not available. "
                + f"Possible options: {OPTIONS_GROUP_SIZE}"
            )
