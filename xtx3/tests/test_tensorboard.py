"""Verify real event data and ensure telemetry does not mutate the model."""

import time

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from xtx3.engine.checkpoint import ModelEMA
from xtx3.engine.trainer import Trainer
from xtx3.losses import XTX3Loss
from xtx3.models import build_model
from xtx3.utils.tensorboard import TrainingMonitor


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_graph_preserves_model_and_records_metadata(tmp_path):
    model = build_model(variant="u").train()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    monitor = TrainingMonitor(tmp_path, {}, model, 64, 7)
    monitor.close()
    assert model.training and all(module.training for module in model.modules())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    events = EventAccumulator(str(monitor.log_dir)).Reload()
    assert events.Tags()["graph"]
    assert events.Graph().node
    assert {"model/architecture/text_summary", "model/parameters/text_summary", "run/config/text_summary"} <= set(events.Tags()["tensors"])
    assert events.Scalars("model/parameter_count")[0].step == 7


def test_training_epoch_writes_real_losses_and_gradients(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.config = {"training": {"tensorboard": {"graph": False}}}
    trainer.output_dir = tmp_path
    trainer.model = build_model(variant="u")
    trainer.criterion = XTX3Loss.from_config({})
    trainer.optimizer = torch.optim.SGD(list(trainer.model.parameters()) + list(trainer.criterion.parameters()), lr=0.01)
    trainer.scaler = torch.amp.GradScaler(enabled=False)
    trainer.ema = ModelEMA(trainer.model)
    trainer.device = "cpu"
    trainer.network_size = 64
    trainer.multi_scale = []
    trainer.use_amp = False
    trainer.epochs = 1
    trainer.batch_size = 2
    trainer.lr0 = 0.01
    trainer.lr_final_factor = 0.01
    trainer.warmup_epochs = 0
    trainer.ckpt_interval_min = 1000
    trainer.global_step = 12
    target = {"boxes": torch.tensor([[8., 8., 56., 56.]]),
              "keypoints": torch.tensor([[[32., 32., 2.]] * 9])}
    trainer.train_loader = [(torch.rand(2, 3, 64, 64), [target, target], None, None)]
    trainer.monitor = TrainingMonitor(tmp_path, trainer.config, trainer.model, 64, trainer.global_step)
    trainer._train_epoch(0, time.time())
    trainer.monitor.scalars("validation", {"AP": 0.5, "AP50": 0.7}, trainer.global_step)
    trainer.monitor.close()
    events = EventAccumulator(str(trainer.monitor.log_dir)).Reload()
    assert events.Scalars("loss/total")[0].step == 13
    assert events.Scalars("training/gradient_norm_before_clip")[0].value > 0
    assert events.Scalars("validation/AP")[0].value == 0.5
    assert any(tag.startswith("weights/model/") for tag in events.Tags()["histograms"])
    assert any(tag.startswith("gradients/model/") for tag in events.Tags()["histograms"])


def test_session_is_unique_and_writer_closes_on_interrupt(tmp_path, monkeypatch):
    trainer = Trainer.__new__(Trainer)
    trainer.config = {"training": {"tensorboard": {"graph": False}}}
    trainer.output_dir = tmp_path
    trainer.model = build_model(variant="u")
    trainer.network_size = 64
    trainer.global_step = 5
    first = TrainingMonitor(tmp_path, trainer.config, trainer.model, 64, 5)
    first.close()

    def interrupted():
        trainer.monitor.scalars("loss", {"total": 1.25}, 6)
        raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "_train_loop", interrupted)
    with pytest.raises(KeyboardInterrupt):
        trainer.train()
    assert trainer.monitor.log_dir != first.log_dir
    assert trainer.monitor.writer.file_writer is None
    events = EventAccumulator(str(trainer.monitor.log_dir)).Reload()
    assert events.Scalars("loss/total")[0].value == 1.25
