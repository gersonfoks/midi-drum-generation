"""Training callbacks: a per-epoch generation-statistics logger and an Optuna
pruning hook."""

from __future__ import annotations

import lightning as pl
import optuna
import torch


class GrooveSampleCallback(pl.Callback):
    """Each validation epoch, sample a few grooves and log a cheap sanity metric.

    Generates ``num_samples`` sequences (split across beats and fills, cycling
    through genres) and logs ``val_mean_gen_length`` — the mean number of tokens
    emitted before the first ``EOS`` — to TensorBoard. It is a lightweight, audio-
    free health signal that the model is learning to end sequences, complementing
    the loss/accuracy scalars Lightning already logs.
    """

    def __init__(
        self,
        bos_token_id: int,
        eos_token_id: int,
        genre_vocab: int,
        max_length: int,
        num_samples: int = 8,
        temperature: float = 1.0,
    ):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.genre_vocab = genre_vocab
        self.max_length = max_length
        self.num_samples = num_samples
        self.temperature = temperature

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        # Cycle genres; alternate beats/fills so the metric reflects both.
        genre_ids = torch.tensor(
            [i % self.genre_vocab for i in range(self.num_samples)], dtype=torch.long
        )
        fill_ids = torch.tensor(
            [i % 2 for i in range(self.num_samples)], dtype=torch.long
        )
        batch = pl_module.generate_batch(  # type: ignore[attr-defined]
            self.max_length,
            genre_ids,
            fill_ids,
            self.bos_token_id,
            temperature=self.temperature,
        )

        lengths = []
        for row in batch.tolist():
            if self.eos_token_id in row:
                lengths.append(row.index(self.eos_token_id))
            else:
                lengths.append(self.max_length)
        mean_length = sum(lengths) / len(lengths)
        pl_module.log("val_mean_gen_length", float(mean_length), prog_bar=True)


class _PruningCallback(pl.Callback):
    """Report ``val_loss`` to an Optuna trial and prune unpromising runs.

    A lightweight stand-in for ``optuna.integration.PyTorchLightningPruningCallback``
    so we don't depend on the separately-versioned ``optuna-integration`` package.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        epoch = trainer.current_epoch
        self.trial.report(float(value), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"pruned at epoch {epoch} ({self.monitor}={float(value):.4f})"
            )
