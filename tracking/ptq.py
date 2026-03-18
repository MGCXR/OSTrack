import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run PTQ from tracking interface.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/test/vitb_256_mae_ce_32x4_ep300_hybrid_hw_mf_co.yaml",
        help="Path to yaml config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="convert/test_mf/models/output.onnx",
        help="Path to checkpoint file or directory.",
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
        help="Also save quantized state_dict.",
    )
    parser.add_argument(
        "--state-dict-output",
        type=str,
        default="",
        help="Path to save quantized state_dict.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization mode.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["fbgemm", "qnnpack"],
        default="fbgemm",
        help="Static PTQ backend.",
    )
    parser.add_argument(
        "--calib-data",
        type=str,
        default="",
        help="Calibration file path for static PTQ.",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=16,
        help="Calibration batch count for static PTQ.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=20,
        help="Warmup iterations for benchmark.",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=80,
        help="Benchmark iterations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Input batch size for check/benchmark/calibration.",
    )
    parser.add_argument(
        "--print-cmd",
        action="store_true",
        help="Print the final command before execution.",
    )
    return parser.parse_args()


def build_ptq_command(args, ptq_script):
    cmd = [
        sys.executable,
        ptq_script,
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--output",
        args.output,
        "--mode",
        args.mode,
        "--backend",
        args.backend,
        "--calib-batches",
        str(args.calib_batches),
        "--warmup-iters",
        str(args.warmup_iters),
        "--bench-iters",
        str(args.bench_iters),
        "--batch-size",
        str(args.batch_size),
    ]

    if args.calib_data != "":
        cmd.extend(["--calib-data", args.calib_data])
    if args.save_state_dict:
        cmd.append("--save-state-dict")
    if args.state_dict_output != "":
        cmd.extend(["--state-dict-output", args.state_dict_output])

    return cmd


def main():
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ptq_script = os.path.join(project_root, "ptq", "run_ptq.py")

    if not os.path.isfile(ptq_script):
        raise FileNotFoundError(f"PTQ script not found: {ptq_script}")

    cmd = build_ptq_command(args, ptq_script)

    if args.print_cmd:
        print("Executing:", " ".join(cmd))

    completed = subprocess.run(cmd, cwd=project_root)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
