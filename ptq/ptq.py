import argparse
import os
import sys

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from lib.config.test.config import cfg, update_config_from_file
from lib.models.test import build_test
from ptq.unit import (
    benchmark_latency_ms,
    bytes_of_torch_obj,
    default_state_dict_output_path,
    format_mb,
    load_calibration_batches,
    quantize_dynamic_linear,
    quantize_static_fx,
    quick_forward_check,
    save_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser("PTQ for OSTrack Test model")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/test/vitb_256_mae_ce_32x4_ep300_hybrid_hw_mf_co.yaml",
        help="Path to yaml config, e.g. experiments/test/vitb_256_mae_ce_32x4_ep300_hybrid_mf.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/checkpoints/train/test/vitb_256_mae_ce_32x4_ep300_hybrid_hw_mf_co",
        help="Path to float checkpoint file (.pth/.pth.tar) or checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ptq/quantized_test_dynamic.pth",
        help="Path to save quantized model object.",
    )
    parser.add_argument(
        "--save-state-dict",
        action="store_true",
        help="If set, also save quantized state_dict.",
    )
    parser.add_argument(
        "--state-dict-output",
        type=str,
        default="",
        help="Path to save quantized state_dict. Default: <output>_state_dict.pth",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization mode. static is experimental for this architecture.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["fbgemm", "qnnpack"],
        default="fbgemm",
        help="Quantization backend for static PTQ.",
    )
    parser.add_argument(
        "--calib-data",
        type=str,
        default="",
        help=(
            "Optional .pt calibration file. Supported formats: "
            "list[dict], dict[str, Tensor], list/tuple of tensors."
        ),
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=16,
        help="Number of calibration batches for static PTQ.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=20,
        help="Warmup iterations for latency benchmark.",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=80,
        help="Measured iterations for latency benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for forward check / benchmark / random calibration.",
    )
    return parser.parse_args()


def load_float_model(config_path, checkpoint_path):
    update_config_from_file(config_path)
    model = build_test(cfg, training=False)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        raise RuntimeError(
            "Checkpoint key mismatch. "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )

    model.eval()
    return model


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    if os.path.isdir(checkpoint_path):
        candidates = [
            os.path.join(checkpoint_path, name)
            for name in os.listdir(checkpoint_path)
            if name.startswith("Test_ep") and (name.endswith(".pth") or name.endswith(".pth.tar"))
        ]
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No checkpoint files like Test_ep*.pth(.tar) found in directory: {checkpoint_path}"
            )
        candidates.sort(key=os.path.getmtime)
        latest = candidates[-1]
        print(f"Resolved checkpoint directory to latest file: {latest}")
        return latest

    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")


def main():
    args = parse_args()

    config_path = os.path.abspath(args.config)
    checkpoint_path = resolve_checkpoint_path(os.path.abspath(args.checkpoint))
    output_path = os.path.abspath(args.output)
    calib_path = os.path.abspath(args.calib_data) if args.calib_data != "" else ""

    if args.save_state_dict and args.state_dict_output.strip() == "":
        state_dict_output_path = default_state_dict_output_path(output_path)
    elif args.state_dict_output.strip() != "":
        state_dict_output_path = os.path.abspath(args.state_dict_output)
    else:
        state_dict_output_path = ""

    print(f"[1/5] Load float model from: {checkpoint_path}")
    model_fp32 = load_float_model(config_path, checkpoint_path)

    print("[2/5] Float model forward check...")
    fp32_shape = quick_forward_check(model_fp32, args.batch_size)
    print(f"      pred_boxes shape(fp32): {fp32_shape}")

    print("[3/5] Apply PTQ...")
    if args.mode == "dynamic":
        model_int8 = quantize_dynamic_linear(model_fp32)
        quant_name = "dynamic_int8_linear"
    else:
        calibration_batches = load_calibration_batches(
            calib_path,
            calib_batches=args.calib_batches,
            batch_size=args.batch_size,
        )
        print(f"      static calibration batches: {len(calibration_batches)}")
        model_int8 = quantize_static_fx(
            model_fp32,
            backend=args.backend,
            calibration_batches=calibration_batches,
        )
        quant_name = f"static_fx_int8_{args.backend}"

    print("[4/5] Quantized model forward check + benchmark...")
    int8_shape = quick_forward_check(model_int8, args.batch_size)
    print(f"      pred_boxes shape(int8): {int8_shape}")

    fp32_latency = benchmark_latency_ms(
        model_fp32,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    int8_latency = benchmark_latency_ms(
        model_int8,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    speedup = fp32_latency / max(int8_latency, 1e-9)
    print(f"      latency fp32: {fp32_latency:.3f} ms")
    print(f"      latency int8: {int8_latency:.3f} ms")
    print(f"      speedup: {speedup:.3f}x")

    fp32_size_bytes = bytes_of_torch_obj(model_fp32.state_dict())
    int8_size_bytes = bytes_of_torch_obj(model_int8.state_dict())
    compress = fp32_size_bytes / max(int8_size_bytes, 1)
    print(f"      state_dict size fp32: {format_mb(fp32_size_bytes)}")
    print(f"      state_dict size int8: {format_mb(int8_size_bytes)}")
    print(f"      compression ratio: {compress:.3f}x")

    print("[5/5] Save outputs...")
    meta = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "quantization": quant_name,
        "test_type": cfg.TEST.TYPE,
        "batch_size": args.batch_size,
        "benchmark": {
            "fp32_latency_ms": fp32_latency,
            "int8_latency_ms": int8_latency,
            "speedup": speedup,
        },
        "size": {
            "fp32_state_dict_bytes": fp32_size_bytes,
            "int8_state_dict_bytes": int8_size_bytes,
            "compression_ratio": compress,
        },
    }

    save_outputs(
        output_path=output_path,
        state_dict_output_path=state_dict_output_path,
        save_state_dict=args.save_state_dict,
        model=model_int8,
        meta=meta,
    )


if __name__ == "__main__":
    main()