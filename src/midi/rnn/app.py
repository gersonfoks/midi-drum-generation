"""Train an RNN to generate drum grooves.

Orchestration layer for the MIDI-groove RNN experiment, exposing three Typer
commands (wired into the CLI at ``src/midi/cli.py`` as ``midi rnn ...``):

* :func:`train` — a config-driven training run (:class:`TrainConfig`, loadable
  from YAML via :meth:`TrainConfig.from_yaml`).
* :func:`smoke_test` — a tiny end-to-end run that trains one step and decodes a
  conditioned sample, used to catch wiring breakage fast.
* :func:`hyperparam_search` — an Optuna study (SQLite-backed, resumable) tuning
  model/optimiser hyperparameters against validation loss.
"""

from __future__ import annotations

from pathlib import Path

import optuna
import typer

from .callbacks import _PruningCallback
from .train import TrainConfig, train_rnn

app = typer.Typer(help="Train an RNN to generate drum grooves.")


def _smoke_test() -> None:
    """Tiny end-to-end run: train one step, sample a conditioned groove.

    Kept fast — a handful of clips, the raw REMI vocab (no BPE training),
    ``fast_dev_run`` — so it can guard the whole pipeline without a real training
    budget. Requires the Groove dataset to be present under ``data_root``.
    """
    config = TrainConfig(
        use_bpe=False,
        limit_per_split=8,
        max_seq_len=4096,
        batch_size=4,
        max_epochs=1,
        hidden_size=16,
        embedding_dim=8,
        genre_embedding_dim=4,
        fill_embedding_dim=4,
        n_layers=1,
        num_samples=4,
        accelerator="cpu",
        log_audio=False,  # keep the smoke test fast and free of the fluidsynth dep
    )
    result = train_rnn(config, trainer_overrides={"fast_dev_run": True})
    model = result["model"]
    tokenizer = result["tokenizer"]

    bos = tokenizer.vocab["BOS_None"]
    tokens = model.generate(length=32, genre_id=0, fill_id=1, bos_token_id=bos)
    assert len(tokens) == 32, "generation returned wrong length"

    print(
        "smoke test passed: trained one step and sampled a "
        f"{len(tokens)}-token groove (vocab_size={len(tokenizer)})."
    )


def _hyperparameter_search(
    n_trials: int = 20,
    study_name: str = "midi_rnn",
    base_config: TrainConfig | None = None,
    *,
    epochs_per_trial: int = 5,
    storage_dir: str | Path = "optuna_studies",
) -> optuna.Study:
    """Tune hyperparameters with Optuna against validation loss.

    The study is SQLite-backed (``<storage_dir>/<study_name>.db``) and resumable:
    re-running with the same ``study_name`` continues where it left off. Each trial
    samples model/optimiser hyperparameters, trains for ``epochs_per_trial``
    epochs, and reports ``val_loss`` (lower is better). A median pruner stops
    clearly-losing trials early via :class:`_PruningCallback`.
    """
    base = base_config or TrainConfig()
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_dir / f'{study_name}.db'}"

    def objective(trial: optuna.Trial) -> float:
        config = base.model_copy(
            update=dict(
                cell_type=trial.suggest_categorical("cell_type", ["rnn", "lstm", "gru"]),
                embedding_dim=trial.suggest_categorical("embedding_dim", [32, 64, 128]),
                hidden_size=trial.suggest_categorical("hidden_size", [64, 128, 256]),
                n_layers=trial.suggest_int("n_layers", 2, 5),
                residual=trial.suggest_categorical("residual", [True, False]),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
                dropout=trial.suggest_float("dropout", 0.0, 0.5),
                grad_clip=trial.suggest_categorical("grad_clip", [0.0, 0.5, 1.0, 5.0]),
                max_epochs=epochs_per_trial,
                log_audio=False,  # don't synthesise audio on every tuning trial
            )
        )
        result = train_rnn(
            config,
            log_samples=False,
            extra_callbacks=[_PruningCallback(trial)],
            trainer_overrides={"enable_progress_bar": False},
        )
        val_loss = result["val_loss"]
        if val_loss is None:
            raise optuna.TrialPruned("no val_loss recorded")
        return val_loss

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest val_loss: {study.best_value:.4f}")
    print(f"Best params:   {study.best_params}")
    print(f"Study stored at: {storage}")
    return study


@app.command()
def train(
    config: Path = typer.Option(
        Path("configs/midi_rnn.yaml"),
        "--config",
        "-c",
        help="Path to a YAML training config.",
    ),
) -> None:
    """Train the RNN model on the drum-groove generation task."""
    cfg = TrainConfig.from_yaml(config)
    result = train_rnn(cfg)
    typer.secho(
        f"Training done. val_loss={result['val_loss']:.4f} val_acc={result['val_acc']:.4f}",
        fg=typer.colors.GREEN,
    )


@app.command()
def smoke_test() -> None:
    """Run a fast end-to-end check of the RNN training pipeline."""
    _smoke_test()


@app.command()
def hyperparam_search(
    n_trials: int = typer.Option(
        20, "--n-trials", "-n", min=1, help="Number of Optuna trials."
    ),
    study_name: str = typer.Option("midi_rnn", "--study-name", help="Optuna study name."),
    config: Path = typer.Option(
        Path("configs/midi_rnn.yaml"),
        "--config",
        "-c",
        help="Base YAML config to vary from.",
    ),
    epochs_per_trial: int = typer.Option(
        200, "--epochs-per-trial", help="Epochs each trial trains."
    ),
) -> None:
    """Search RNN hyperparameters with Optuna (SQLite-backed, resumable)."""
    base = TrainConfig.from_yaml(config)
    _hyperparameter_search(
        n_trials=n_trials,
        study_name=study_name,
        base_config=base,
        epochs_per_trial=epochs_per_trial,
    )


if __name__ == "__main__":
    app()
