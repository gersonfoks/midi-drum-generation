from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import lightning as pl
import yaml
from pydantic import BaseModel, field_validator

from lightning.pytorch.loggers import TensorBoardLogger

from src.maze.rnn.callbacks import MazeSampleCallback

from ..dataset import VOCAB_SIZE, build_dataloaders
from ..maze import CarveStrategy
from .helpers import sample_maze, sample_mazes
from .lightning_model import LightningRnnModel
from .model import RNNModel


class TrainConfig(BaseModel):
    """All knobs for a training run. Every field has a sensible default so a
    partial YAML file (or none at all) still produces a valid config."""

    # Data
    maze_size: int = 11
    carve_strategy: CarveStrategy = CarveStrategy.random
    num_train_samples: int = 5000
    # Model
    cell_type: Literal["rnn", "lstm", "gru"] = "rnn"
    embedding_dim: int = 64
    hidden_size: int = 128
    n_layers: int = 5

    # Optimisation
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 20

    # Runtime
    seed: int = 0
    accelerator: str = "auto"
    num_workers: int = 0

    # Sampling / logging
    sample_temperature: float = 1.0
    num_vis_samples: int = 16
    num_valid_samples: int = 64
    log_dir: str = "lightning_logs"

    @field_validator("maze_size")
    @classmethod
    def _maze_size_odd(cls, value: int) -> int:
        if value < 3 or value % 2 == 0:
            raise ValueError(f"maze_size must be an odd number >= 3, got {value}")
        return value

    @property
    def sequence_length(self) -> int:
        """Number of cell tokens per maze (the flattened grid, excluding BOS)."""
        return self.maze_size * self.maze_size

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load a config from a YAML file, merging its keys over the defaults."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)


def build_model(config: TrainConfig) -> LightningRnnModel:
    """Construct the RNN and its Lightning wrapper from ``config``."""
    model = RNNModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        n_layers=config.n_layers,
        cell_type=config.cell_type,
    )
    return LightningRnnModel(model, learning_rate=config.learning_rate)


CellType = Literal["rnn", "lstm", "gru"]
def train_rnn(
    config: TrainConfig,
    *,
    log_samples: bool = True,
    extra_callbacks: list[pl.Callback] | None = None,
    trainer_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a training loop for ``config`` and return the final metrics + model.

    When ``log_samples`` is set, a :class:`MazeSampleCallback` logs generated-maze
    images and a ``val_valid_fraction`` metric each epoch. ``extra_callbacks`` and
    ``trainer_overrides`` are hooks used by :func:`hyperparameter_search` to inject
    a pruning callback and shrink the run.
    """
    pl.seed_everything(config.seed, verbose=False)
    train_loader, val_loader, _ = build_dataloaders(
        maze_size=config.maze_size,
        carve_strategy=config.carve_strategy,
        num_train_samples=config.num_train_samples,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    model = build_model(config)

    callbacks: list[pl.Callback] = list(extra_callbacks or [])
    if log_samples:
        callbacks.append(
            MazeSampleCallback(
                config.maze_size,
                config.num_valid_samples,
                config.num_vis_samples,
                config.sample_temperature,
            )
        )

    trainer_kwargs: dict[str, Any] = dict(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=config.log_dir, name="maze_rnn"),
        enable_checkpointing=False,
    )
    trainer_kwargs.update(trainer_overrides or {})
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader)

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    return {
        "val_loss": metrics.get("val_loss"),
        "val_acc": metrics.get("val_acc"),
        "val_valid_fraction": metrics.get("val_valid_fraction"),
        "metrics": metrics,
        "model": model,
        "trainer": trainer,
    }


# ``sample_maze`` / ``sample_mazes`` live in ``helpers`` and are re-exported here so
# callers importing them from ``train`` keep working.
__all__ = ["TrainConfig", "build_model", "train_rnn", "sample_maze", "sample_mazes"]