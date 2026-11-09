class QuantProperties:
    """Quantization requires group size, bits, and mode."""

    mode_options = ["affine"]

    def __init__(self, group_size: int = 64, bits: int = 8, mode: str = "affine"):
        self._group_size: int = group_size
        self._bits: int = bits
        self._mode: str = mode
        self._validate_mode(mode)

    def _validate_mode(self, mode: str):
        if mode not in self.mode_options:
            raise ValueError(
                f"Quantizing with mode {mode} is not available. "
                + f"Possible options: {self.mode_options}"
            )
