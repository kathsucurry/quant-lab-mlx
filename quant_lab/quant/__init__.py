from ..models import load_model
from .config import QuantProperties


def quantize_model(
    model_path: str,
    model_quant_path: str,
    group_size: int = 64,
    bits: int = 8,
    mode: str = "affine",
) -> None:
    model, model_config = load_model(model_path)

    quant_props = QuantProperties(group_size, bits, mode)

    print("Successful load!")
