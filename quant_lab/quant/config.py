OPTIONS_GROUP_SIZE = [16, 32, 64, 128]
OPTIONS_MODE = ["affine"]


class QuantProperties:
    """Quantization requires group size, bits, and mode."""

    def __init__(self, group_size: int = 64, bits: int = 8, mode: str = "affine"):
        self.group_size: int = group_size
        self.bits: int = bits
        self.mode: str = mode
        self._validate_properties()

    def _validate_properties(self):
        if self.mode not in OPTIONS_MODE:
            raise ValueError(
                f"Quantizing with mode {self.mode} is not available. "
                + f"Possible options: {OPTIONS_MODE}"
            )

        if not isinstance(self.group_size, int):
            raise ValueError("Group size must be an integer.")

        if self.group_size not in OPTIONS_GROUP_SIZE:
            raise ValueError(
                f"Quantizing with group size {self.group_size} is not available. "
                + f"Possible options: {OPTIONS_GROUP_SIZE}"
            )
