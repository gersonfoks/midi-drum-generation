"""Config-driven training for the conditional MIDI-groove RNN."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import lightning as pl
import yaml
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel, field_validator

from ..callbacks.generate_wav_callback import GenerateWavCallback
from ..data.groove_dataset import GrooveDataset
from .callbacks import GrooveSampleCallback
from .dataset import (
    FILL_VOCAB,
    GENRE_VOCAB,
    build_dataloaders,
    build_tokenizer,
)
from .lightning_model import LightningRnnModel
from .model import RNNModel

CellType = Literal["rnn", "lstm", "gru"]


class TrainConfig(BaseModel):
    """All knobs for a training run. Every field has a sensible default so a
    partial YAML file (or none at all) still produces a valid config."""

    # Data
    data_root: str = "data/raw/groove"
    max_seq_len: int = 1024
    vocab_size: int = 1000
    use_bpe: bool = True
    tokenizer_path: str = "artifacts/midi_tokenizer.json"
    # Cap the number of clips per split (before length filtering). Handy for a
    # fast smoke test; ``None`` uses the whole dataset.
    limit_per_split: int | None = None

    # Model
    cell_type: CellType = "lstm"
    embedding_dim: int = 64
    genre_embedding_dim: int = 16
    fill_embedding_dim: int = 8
    hidden_size: int = 256
    n_layers: int = 4
    dropout: float = 0.1
    # Add a skip connection around each recurrent layer. Eases gradient flow
    # through a deep stack, helping on longer sequences.
    residual: bool = False

    # Optimisation
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    # Clip gradients to this global L2 norm (0 disables). Curbs exploding
    # gradients so longer sequences train stably.
    grad_clip: float = 1.0

    # Runtime
    seed: int = 0
    accelerator: str = "auto"
    num_workers: int = 0

    # Sampling / logging
    sample_temperature: float = 1.0
    num_samples: int = 8
    log_dir: str = "lightning_logs"
    log_name: str = "midi_rnn"

    # Audio samples (synthesise generated grooves so you can listen to them)
    log_audio: bool = True
    audio_every_n_epochs: int = 5
    audio_gen_length: int = 512
    audio_sample_rate: int = 44100
    audio_temperature: float = 1.0

    @field_validator("max_seq_len")
    @classmethod
    def _max_seq_len_min(cls, value: int) -> int:
        if value < 2:
            raise ValueError(f"max_seq_len must be >= 2, got {value}")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load a config from a YAML file, merging its keys over the defaults."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)


def build_model(config: TrainConfig, vocab_size: int, pad_token_id: int) -> LightningRnnModel:
    """Construct the RNN and its Lightning wrapper from ``config``."""
    model = RNNModel(
        vocab_size=vocab_size,
        genre_vocab=GENRE_VOCAB,
        fill_vocab=FILL_VOCAB,
        embedding_dim=config.embedding_dim,
        genre_embedding_dim=config.genre_embedding_dim,
        fill_embedding_dim=config.fill_embedding_dim,
        hidden_size=config.hidden_size,
        n_layers=config.n_layers,
        cell_type=config.cell_type,
        pad_token_id=pad_token_id,
        dropout=config.dropout,
        residual=config.residual,
    )
    return LightningRnnModel(
        model,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        learning_rate=config.learning_rate,
    )


def train_rnn(
    config: TrainConfig,
    *,
    log_samples: bool = True,
    extra_callbacks: list[pl.Callback] | None = None,
    trainer_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a training loop for ``config`` and return the final metrics + model.

    Builds the tokenizer (BPE-trained/cached when ``config.use_bpe``), the Groove
    dataloaders, and the model, then fits with a :class:`TensorBoardLogger`. When
    ``log_samples`` is set, a :class:`GrooveSampleCallback` logs a
    ``val_mean_gen_length`` metric each epoch. ``extra_callbacks`` and
    ``trainer_overrides`` are hooks used by the Optuna search to inject a pruning
    callback and shrink each run.
    """
    pl.seed_everything(config.seed, verbose=False)

    # BPE (when enabled) trains on the train split only, so the vocabulary never
    # sees validation/test performances.
    train_points = GrooveDataset(config.data_root).split("train")
    tokenizer = build_tokenizer(
        vocab_size=config.vocab_size,
        use_bpe=config.use_bpe,
        dataset=train_points,
        cache_path=config.tokenizer_path,
    )
    train_loader, val_loader, _ = build_dataloaders(config, tokenizer)
    vocab_size = len(tokenizer)
    model = build_model(config, vocab_size, tokenizer.pad_token_id)

    logger = TensorBoardLogger(save_dir=config.log_dir, name=config.log_name)

    callbacks: list[pl.Callback] = list(extra_callbacks or [])
    if log_samples:
        callbacks.append(
            GrooveSampleCallback(
                bos_token_id=tokenizer.vocab["BOS_None"],
                eos_token_id=tokenizer.vocab["EOS_None"],
                genre_vocab=GENRE_VOCAB,
                max_length=config.max_seq_len,
                num_samples=config.num_samples,
                temperature=config.sample_temperature,
            )
        )
    if config.log_audio:
        callbacks.append(
            GenerateWavCallback(
                tokenizer=tokenizer,
                bos_token_id=tokenizer.vocab["BOS_None"],
                gen_length=config.audio_gen_length,
                temperature=config.audio_temperature,
                sample_rate=config.audio_sample_rate,
                every_n_epochs=config.audio_every_n_epochs,
                out_dir=Path(logger.log_dir) / "audio",
                tag_prefix="samples",
            )
        )

    trainer_kwargs: dict[str, Any] = dict(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=False,
        # Lightning clips the global grad norm before each optimiser step; None
        # disables it.
        gradient_clip_val=config.grad_clip if config.grad_clip > 0 else None,
    )
    trainer_kwargs.update(trainer_overrides or {})
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_loader)

    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    return {
        "val_loss": metrics.get("val_loss"),
        "val_acc": metrics.get("val_acc"),
        "val_mean_gen_length": metrics.get("val_mean_gen_length"),
        "metrics": metrics,
        "model": model,
        "tokenizer": tokenizer,
        "trainer": trainer,
    }


__all__ = ["TrainConfig", "CellType", "build_model", "train_rnn"]
