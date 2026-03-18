import argparse
import os
import time
from typing import Dict, List

import numpy as np
import onnx

try:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
except ImportError as exc:
    raise ImportError(
        "onnxruntime and onnxruntime-tools are required for ONNX PTQ. "
        "Please install onnxruntime (or onnxruntime-gpu)."
    ) from exc


class NumpyCalibrationReader(CalibrationDataReader):
    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        self.samples = samples
        self.index = 0

    def get_next(self):
        if self.index >= len(self.samples):
            return None
        sample = self.samples[self.index]
        self.index += 1
        return sample


def parse_args():
    parser = argparse.ArgumentParser("ONNX-only PTQ for OSTrack model")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/test/vitb_256_mae_ce_32x4_ep300_hybrid_hw_mf_co.yaml",
        help="Reserved for compatibility. Not used in ONNX-only mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="convert/test_mf/models/output.onnx",
        help="Path to ONNX model file or directory containing ONNX files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ptq/quantized_test_dynamic.onnx",
        help="Path to save quantized ONNX model.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization mode for ONNX Runtime.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fbgemm",
        help="Reserved for compatibility. Not used in ONNX-only mode.",
    )
    parser.add_argument(
        "--calib-data",
        type=str,
        default="",
        help="Optional .npz calibration file for static quantization.",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=16,
        help="Calibration sample count for static quantization.",
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
        help="Batch size for random benchmark/calibration inputs.",
    )
    parser.add_argument(
        "--save-state-dict",
        action="store_true",
        help="Reserved for compatibility. Ignored in ONNX-only mode.",
    )
    parser.add_argument(
        "--state-dict-output",
        type=str,
        default="",
        help="Reserved for compatibility. Ignored in ONNX-only mode.",
    )
    return parser.parse_args()


def resolve_onnx_path(path: str) -> str:
    if os.path.isfile(path):
        if not path.lower().endswith(".onnx"):
            raise ValueError(f"Only .onnx is supported in this script: {path}")
        return path

    if os.path.isdir(path):
        candidates = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.lower().endswith(".onnx")
        ]
        if len(candidates) == 0:
            raise FileNotFoundError(f"No .onnx model found in directory: {path}")
        candidates.sort(key=os.path.getmtime)
        latest = candidates[-1]
        print(f"Resolved ONNX directory to latest file: {latest}")
        return latest

    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def create_session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def normalize_dim(dim, fallback: int) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    return fallback


def random_input_from_session(session: ort.InferenceSession, batch_size: int) -> Dict[str, np.ndarray]:
    feed = {}
    for i_meta in session.get_inputs():
        shape = []
        for idx, dim in enumerate(i_meta.shape):
            if idx == 0:
                shape.append(normalize_dim(dim, batch_size))
            else:
                shape.append(normalize_dim(dim, 1))

        if "float16" in i_meta.type:
            arr = np.random.randn(*shape).astype(np.float16)
        else:
            arr = np.random.randn(*shape).astype(np.float32)
        feed[i_meta.name] = arr
    return feed


def load_npz_calibration(calib_path: str, input_names: List[str], limit: int) -> List[Dict[str, np.ndarray]]:
    data = np.load(calib_path, allow_pickle=True)
    if not all(name in data for name in input_names):
        missing = [name for name in input_names if name not in data]
        raise KeyError(
            "Calibration npz must contain one array per ONNX input name. "
            f"Missing keys: {missing}"
        )

    total = int(data[input_names[0]].shape[0])
    count = min(total, limit)
    samples = []
    for i in range(count):
        sample = {name: np.asarray(data[name][i : i + 1]) for name in input_names}
        samples.append(sample)
    return samples


def build_calibration_samples(session: ort.InferenceSession, calib_path: str, calib_batches: int, batch_size: int):
    input_names = [x.name for x in session.get_inputs()]
    if calib_path == "":
        return [random_input_from_session(session, batch_size) for _ in range(calib_batches)]

    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"Calibration file does not exist: {calib_path}")
    if not calib_path.lower().endswith(".npz"):
        raise ValueError("Only .npz calibration file is supported in ONNX-only mode.")

    samples = load_npz_calibration(calib_path, input_names, calib_batches)
    if len(samples) == 0:
        raise ValueError("No calibration samples loaded from npz file.")
    return samples


def benchmark_latency_ms(session: ort.InferenceSession, batch_size: int, warmup_iters: int, bench_iters: int) -> float:
    out_names = [o.name for o in session.get_outputs()]
    for _ in range(warmup_iters):
        session.run(out_names, random_input_from_session(session, batch_size))

    t0 = time.perf_counter()
    for _ in range(bench_iters):
        session.run(out_names, random_input_from_session(session, batch_size))
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / max(bench_iters, 1)


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024.0 * 1024.0)


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent != "":
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()

    model_path = resolve_onnx_path(os.path.abspath(args.checkpoint))
    output_path = os.path.abspath(args.output)
    calib_path = os.path.abspath(args.calib_data) if args.calib_data != "" else ""

    ensure_parent_dir(output_path)

    print(f"[1/5] Load and check ONNX model: {model_path}")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("      ONNX model check passed")

    print("[2/5] Build runtime session and baseline benchmark...")
    fp32_sess = create_session(model_path)
    fp32_latency = benchmark_latency_ms(
        fp32_sess,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    fp32_size = file_size_mb(model_path)

    print("[3/5] Apply ONNX quantization...")
    if args.mode == "dynamic":
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
        )
        quant_name = "onnx_dynamic_int8"
    else:
        calib_samples = build_calibration_samples(
            fp32_sess,
            calib_path=calib_path,
            calib_batches=args.calib_batches,
            batch_size=args.batch_size,
        )
        print(f"      static calibration samples: {len(calib_samples)}")
        reader = NumpyCalibrationReader(calib_samples)
        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )
        quant_name = "onnx_static_int8_qdq"

    print("[4/5] Quantized benchmark...")
    int8_sess = create_session(output_path)
    int8_latency = benchmark_latency_ms(
        int8_sess,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    int8_size = file_size_mb(output_path)

    speedup = fp32_latency / max(int8_latency, 1e-9)
    ratio = fp32_size / max(int8_size, 1e-9)

    print(f"      latency fp32: {fp32_latency:.3f} ms")
    print(f"      latency int8: {int8_latency:.3f} ms")
    print(f"      speedup: {speedup:.3f}x")
    print(f"      model size fp32: {fp32_size:.2f} MB")
    print(f"      model size int8: {int8_size:.2f} MB")
    print(f"      compression ratio: {ratio:.3f}x")

    print("[5/5] Done")
    print(f"      quantization: {quant_name}")
    print(f"      output: {output_path}")

    if args.save_state_dict or args.state_dict_output != "":
        print("      note: --save-state-dict/--state-dict-output are ignored in ONNX-only mode")


if __name__ == "__main__":
    main()