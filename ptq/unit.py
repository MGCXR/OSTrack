import io
import os
import time
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.ao.quantization import QConfigMapping, get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from lib.config.test.config import cfg


TensorBatch = Tuple[torch.Tensor, ...]


class SinglePredBoxesWrapper(nn.Module):
	def __init__(self, model: nn.Module):
		super().__init__()
		self.model = model

	def forward(self, template: torch.Tensor, search: torch.Tensor):
		return self.model(template=template, search=search)["pred_boxes"]


class HybridPredBoxesWrapper(nn.Module):
	def __init__(self, model: nn.Module):
		super().__init__()
		self.model = model

	def forward(
		self,
		template: torch.Tensor,
		search: torch.Tensor,
		template_event: torch.Tensor,
		search_event: torch.Tensor,
	):
		return self.model(
			template=template,
			search=search,
			template_event=template_event,
			search_event=search_event,
		)["pred_boxes"]


def make_random_batch(batch_size: int) -> TensorBatch:
	template_size = int(cfg.TEST.TEMPLATE_SIZE)
	search_size = int(cfg.TEST.SEARCH_SIZE)
	template = torch.randn(batch_size, 3, template_size, template_size)
	search = torch.randn(batch_size, 3, search_size, search_size)

	if cfg.TEST.TYPE == "HYBRID":
		template_event = torch.randn(batch_size, 3, template_size, template_size)
		search_event = torch.randn(batch_size, 3, search_size, search_size)
		return (template, search, template_event, search_event)
	return (template, search)


def ensure_batched(t: torch.Tensor) -> torch.Tensor:
	if t.dim() == 3:
		return t.unsqueeze(0)
	return t


def unpack_to_batch(item) -> TensorBatch:
	if isinstance(item, dict):
		if cfg.TEST.TYPE == "HYBRID":
			keys = ["template", "search", "template_event", "search_event"]
		else:
			keys = ["template", "search"]
		return tuple(ensure_batched(item[k]).float() for k in keys)

	if isinstance(item, (list, tuple)):
		tensors = tuple(ensure_batched(x).float() for x in item)
		expected = 4 if cfg.TEST.TYPE == "HYBRID" else 2
		if len(tensors) != expected:
			raise ValueError(f"Calibration sample expects {expected} tensors, got {len(tensors)}")
		return tensors

	raise TypeError("Unsupported calibration sample type.")


def load_calibration_batches(calib_path: str, calib_batches: int, batch_size: int) -> List[TensorBatch]:
	if calib_path == "":
		return [make_random_batch(batch_size) for _ in range(calib_batches)]

	raw = torch.load(calib_path, map_location="cpu", weights_only=False)
	batches: List[TensorBatch] = []

	if isinstance(raw, list):
		for item in raw:
			batches.append(unpack_to_batch(item))
	elif isinstance(raw, tuple):
		if len(raw) > 0 and all(isinstance(x, torch.Tensor) for x in raw):
			batches.append(unpack_to_batch(raw))
		else:
			for item in raw:
				batches.append(unpack_to_batch(item))
	elif isinstance(raw, dict):
		if cfg.TEST.TYPE == "HYBRID":
			keys = ["template", "search", "template_event", "search_event"]
		else:
			keys = ["template", "search"]
		if all(k in raw and isinstance(raw[k], torch.Tensor) for k in keys):
			n = int(raw[keys[0]].shape[0])
			for i in range(n):
				item = tuple(raw[k][i : i + 1].float() for k in keys)
				batches.append(item)
		else:
			batches.append(unpack_to_batch(raw))
	else:
		raise TypeError("Unsupported calibration file format.")

	if len(batches) == 0:
		raise ValueError("No calibration batches were loaded.")
	return batches[:calib_batches]


def forward_pred_boxes(model: nn.Module, batch: TensorBatch) -> torch.Tensor:
	if len(batch) == 4:
		out = model(
			template=batch[0],
			search=batch[1],
			template_event=batch[2],
			search_event=batch[3],
		)
	else:
		out = model(template=batch[0], search=batch[1])

	if isinstance(out, dict):
		if "pred_boxes" not in out:
			raise RuntimeError("Forward check failed: pred_boxes is missing in output dict.")
		return out["pred_boxes"]
	if torch.is_tensor(out):
		return out
	raise RuntimeError("Forward check failed: unsupported output type.")


def quick_forward_check(model: nn.Module, batch_size: int):
	batch = make_random_batch(batch_size)
	with torch.no_grad():
		pred_boxes = forward_pred_boxes(model, batch)
	return tuple(pred_boxes.shape)


def build_static_quant_wrapper(model: nn.Module) -> nn.Module:
	if cfg.TEST.TYPE == "HYBRID":
		return HybridPredBoxesWrapper(model)
	return SinglePredBoxesWrapper(model)


def quantize_dynamic_linear(model: nn.Module) -> nn.Module:
	q_model = torch.ao.quantization.quantize_dynamic(
		model,
		{nn.Linear},
		dtype=torch.qint8,
	)
	q_model.eval()
	return q_model


def quantize_static_fx(
	model: nn.Module,
	backend: str,
	calibration_batches: Sequence[TensorBatch],
) -> nn.Module:
	torch.backends.quantized.engine = backend
	wrapper = build_static_quant_wrapper(model)
	wrapper.eval()

	example_inputs = calibration_batches[0]
	qconfig_mapping = QConfigMapping().set_global(get_default_qconfig(backend))
	prepared = prepare_fx(wrapper, qconfig_mapping, example_inputs=example_inputs)

	with torch.no_grad():
		for batch in calibration_batches:
			prepared(*batch)

	quantized = convert_fx(prepared)
	quantized.eval()
	return quantized


def benchmark_latency_ms(model: nn.Module, batch_size: int, warmup_iters: int, bench_iters: int) -> float:
	model.eval()
	with torch.no_grad():
		for _ in range(warmup_iters):
			forward_pred_boxes(model, make_random_batch(batch_size))

		t0 = time.perf_counter()
		for _ in range(bench_iters):
			forward_pred_boxes(model, make_random_batch(batch_size))
		t1 = time.perf_counter()
	return (t1 - t0) * 1000.0 / max(bench_iters, 1)


def bytes_of_torch_obj(obj) -> int:
	buf = io.BytesIO()
	torch.save(obj, buf)
	return len(buf.getbuffer())


def format_mb(size_bytes: int) -> str:
	return f"{size_bytes / (1024.0 * 1024.0):.2f} MB"


def default_state_dict_output_path(output_path: str) -> str:
	base, ext = os.path.splitext(output_path)
	if ext == "":
		ext = ".pth"
	return base + "_state_dict" + ext


def save_outputs(
	output_path: str,
	state_dict_output_path: str,
	save_state_dict: bool,
	model: nn.Module,
	meta: Dict,
):
	output_dir = os.path.dirname(output_path)
	if output_dir != "":
		os.makedirs(output_dir, exist_ok=True)

	torch.save({"model": model, "meta": meta}, output_path)
	print(f"Saved quantized model object to: {output_path}")

	if save_state_dict:
		state_dir = os.path.dirname(state_dict_output_path)
		if state_dir != "":
			os.makedirs(state_dir, exist_ok=True)
		torch.save({"state_dict": model.state_dict(), "meta": meta}, state_dict_output_path)
		print(f"Saved quantized state_dict to: {state_dict_output_path}")
