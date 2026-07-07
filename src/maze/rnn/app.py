"""Train an RNN to generate mazes.

This module is the orchestration layer for the maze-RNN experiment. It exposes
three entry points, all reachable from the Typer CLI (``src/maze/cli.py``):

* :func:`train` — a config-driven training run. The config is a pydantic
  :class:`TrainConfig` (loadable from YAML via :meth:`TrainConfig.from_yaml`).
* :func:`smoke_test` — a tiny end-to-end run that trains for one step and decodes
  a maze, used to catch wiring breakage fast (in CI and locally).
* :func:`hyperparameter_search` — an Optuna study (SQLite-backed, resumable) that
  tunes model/optimiser hyperparameters against validation loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


import optuna

from src.maze.rnn.callbacks import _PruningCallback
from src.maze.rnn.train import TrainConfig, sample_maze, train_rnn



import typer

app = typer.Typer(help="Train an RNN to generate mazes.")

CellType = Literal["rnn", "lstm", "gru"]


def _smoke_test() -> None:
    """Tiny end-to-end run: train one step, decode a maze, assert shapes line up.

    Kept fast (small maze, a handful of samples, ``fast_dev_run``) so it can guard
    the whole pipeline in CI without a real training budget.
    """
    config = TrainConfig(
        maze_size=5,
        num_train_samples=16,
        batch_size=8,
        max_epochs=1,
        hidden_size=16,
        embedding_dim=8,
        n_layers=1,
        accelerator="cpu",
    )
    result = train_rnn(config, trainer_overrides={"fast_dev_run": True})
    model = result["model"]

    tokens = model.generate(config.sequence_length)
    assert len(tokens) == config.sequence_length, "generation returned wrong length"
    maze, is_proper = sample_maze(model, config.maze_size)
    assert maze.height == config.maze_size and maze.width == config.maze_size
    fraction = result["val_valid_fraction"]
    assert fraction is not None and 0.0 <= fraction <= 1.0, "valid fraction not logged"

    print(
        "smoke test passed: trained one step and decoded a "
        f"{maze.height}x{maze.width} maze (proper={is_proper}, "
        f"val_valid_fraction={fraction:.2f})."
    )


def _hyperparameter_search(
    n_trials: int = 20,
    study_name: str = "maze_rnn",
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
                cell_type=trial.suggest_categorical(
                    "cell_type", ["rnn", "lstm", "gru"]
                ),
                embedding_dim=trial.suggest_categorical("embedding_dim", [32, 64, 128]),
                hidden_size=trial.suggest_categorical("hidden_size", [64, 128, 256]),
                n_layers=trial.suggest_int("n_layers", 1, 3),
                learning_rate=trial.suggest_float(
                    "learning_rate", 1e-4, 1e-2, log=True
                ),
                batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                max_epochs=epochs_per_trial,
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
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to a YAML training config.",
    ),
) -> None:
    """Train the RNN model on the maze generation task."""

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
    study_name: str = typer.Option(
        "maze_rnn", "--study-name", help="Optuna study name."
    ),
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Base YAML config to vary from.",
    ),
    epochs_per_trial: int = typer.Option(
        5, "--epochs-per-trial", help="Epochs each trial trains."
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
