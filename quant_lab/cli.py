"""Command-line interface: ``qlab <command> [options]``."""

import argparse
from collections.abc import Sequence

from .quant import quantize_model


def _quantize(args: argparse.Namespace):
    quantize_model(
        model_path=args.model,
        model_quant_path=args.output_dir,
    )

    # from mlx_lm import load
    # import mlx.core as mx
    # model, tokenizer = load("Qwen/Qwen3-0.6B")

    # mx.quantize(model, group_size=64, bits=8)


def _compute_kl(args: argparse.Namespace):
    # TODO: run args.base and args.quantized on args.data and report the mean KL divergence.
    raise NotImplementedError("qlab compute_kl is not implemented yet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qlab",
        description="Quantize MLX models and measure KL divergence against the original.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    quantize = subparsers.add_parser("quantize", help="Quantize a model.")
    quantize.add_argument("--model", required=True, help="Hugging Face repo id or local path.")
    # quantize.add_argument(
    #     "--type", required=True, dest="quant_type", help="Quantization type, e.g. q4."
    # )
    quantize.add_argument(
        "--output-dir", default=None, help="Where to save. Default: outputs/<model>-<type>."
    )
    quantize.set_defaults(func=_quantize)

    # compute_kl = subparsers.add_parser(
    #     "kl", help="KL divergence between a base and a quantized model."
    # )
    # compute_kl.add_argument("--base", required=True, help="Original model (repo id or path).")
    # compute_kl.add_argument("--quantized", required=True, help="Quantized model path.")
    # compute_kl.add_argument("--data", default="data/eval.txt", help="Eval text file.")
    # compute_kl.set_defaults(func=_compute_kl)

    return parser


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
