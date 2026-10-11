"""Local training telemetry, with one event directory per training session."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch
from torch.utils.tensorboard import SummaryWriter

from .logging import get_logger

logger = get_logger(__name__)


class _InferenceGraph(torch.nn.Module):
    """Expose tensors instead of XTX4's dataclass output to the graph tracer."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        raw = self.model(images).one2one
        return raw.person, raw.kpt


class TrainingMonitor:
    def __init__(self, output_dir: Path, config: dict, model: torch.nn.Module,
                 network_size: int, global_step: int) -> None:
        session = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        self.log_dir = output_dir / "tensorboard" / session
        self.writer = SummaryWriter(str(self.log_dir), flush_secs=5)
        options = config.get("training", {}).get("tensorboard", {})
        self.log_every = max(1, int(options.get("log_every_steps", 10)))
        self.histogram_every = max(1, int(options.get("histogram_every_epochs", 1)))
        try:
            self.writer.add_text("model/architecture", f"```text\n{model}\n```", global_step)
            self.writer.add_text("run/config", f"```json\n{json.dumps(config, indent=2)}\n```", global_step)
            rows = ["| Parameter | Shape | Count | Trainable |", "|---|---|---:|---|"]
            rows.extend(f"| {name} | {tuple(p.shape)} | {p.numel()} | {p.requires_grad} |"
                        for name, p in model.named_parameters())
            self.writer.add_text("model/parameters", "\n".join(rows), global_step)
            self.writer.add_scalar("model/parameter_count", sum(p.numel() for p in model.parameters()), global_step)
            if options.get("graph", True):
                modes = {module: module.training for module in model.modules()}
                try:
                    model.eval()
                    example = next(model.parameters()).new_zeros(1, 3, network_size, network_size)
                    with torch.no_grad():
                        self.writer.add_graph(_InferenceGraph(model).eval(), example)
                except Exception as exc:
                    logger.warning("TensorBoard graph unavailable: %s", exc)
                finally:
                    for module, training in modes.items():
                        module.training = training
            self.writer.flush()
        except BaseException:
            self.writer.close()
            raise
        logger.info("TensorBoard events -> %s", self.log_dir)

    def scalars(self, prefix: str, values: dict, step: int) -> None:
        for name, value in values.items():
            self.writer.add_scalar(f"{prefix}/{name}", float(value), step)

    def histograms(self, modules: dict[str, torch.nn.Module], step: int) -> None:
        # Bounded deterministic samples keep event files and GPU transfers small.
        for prefix, module in modules.items():
            for name, parameter in module.named_parameters():
                for kind, tensor in (("weights", parameter), ("gradients", parameter.grad)):
                    if tensor is None:
                        continue
                    values = tensor.detach().reshape(-1)
                    stride = max(1, (values.numel() + 8191) // 8192)
                    values = values[::stride].float().cpu()
                    finite = torch.isfinite(values)
                    if finite.all() and values.numel():
                        self.writer.add_histogram(f"{kind}/{prefix}/{name}", values, step, bins=64)
                    elif values.numel():
                        self.writer.add_scalar(f"nonfinite/{prefix}/{kind}/{name}", (~finite).sum().item(), step)

    def close(self) -> None:
        self.writer.close()
