from typing import Any

import mlx.nn as nn
from mlx_lm import load


def load_model(model_path: str) -> tuple[nn.Module, dict[str, Any]]:
    model, _, config = load(model_path, lazy=True, return_config=True)
    return model, config
