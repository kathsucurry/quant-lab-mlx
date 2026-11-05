"""Command-line interface: ``qlab <command> [options]``."""

import argparse
from collections.abc import Sequence


def _quantize(args: argparse.Namespace) -> int:
    # TODO: load args.model, quantize it as args.quant_type, save to args.output_dir.
    raise NotImplementedError("qlab quantize is not implemented yet")


def _compute_kl(args: argparse.Namespace) -> int:
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
    quantize.add_argument(
        "--type", required=True, dest="quant_type", help="Quantization type, e.g. q4."
    )
    quantize.add_argument(
        "--output-dir", default=None, help="Where to save. Default: outputs/<model>-<type>."
    )
    quantize.set_defaults(func=_quantize)

    compute_kl = subparsers.add_parser(
        "kl", help="KL divergence between a base and a quantized model."
    )
    compute_kl.add_argument("--base", required=True, help="Original model (repo id or path).")
    compute_kl.add_argument("--quantized", required=True, help="Quantized model path.")
    compute_kl.add_argument("--data", default="data/eval.txt", help="Eval text file.")
    compute_kl.set_defaults(func=_compute_kl)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
