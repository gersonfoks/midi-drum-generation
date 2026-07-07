import lightning as pl
import optuna

from src.maze.maze import MazeVisualizer
from src.maze.rnn.helpers import sample_mazes


class MazeSampleCallback(pl.Callback):
    """Each validation epoch: sample mazes, log the valid fraction, and (if a
    TensorBoard logger is present) log a grid of the first ``num_vis_samples``.

    *Valid* means the decoded grid is a proper maze — connected, no cycles, every
    odd/odd path cell reachable — as verified by ``MazeChecker.check``.
    """

    def __init__(
        self,
        maze_size: int,
        num_valid_samples: int = 64,
        num_vis_samples: int = 16,
        temperature: float = 1.0,
    ):
        self.maze_size = maze_size
        self.num_valid_samples = num_valid_samples
        self.num_vis_samples = num_vis_samples
        self.temperature = temperature

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        n_metric = max(self.num_valid_samples, self.num_vis_samples)
        mazes, valid = sample_mazes(
            pl_module, self.maze_size, n_metric, temperature=self.temperature  # type: ignore[arg-type]
        )

        fraction = sum(valid) / len(valid)
        pl_module.log("val_valid_fraction", fraction, prog_bar=True)

        experiment = getattr(trainer.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_figure"):
            fig = self._make_grid_figure(mazes[: self.num_vis_samples])
            experiment.add_figure("generated_mazes", fig, global_step=trainer.current_epoch)
            import matplotlib.pyplot as plt

            plt.close(fig)

    def _make_grid_figure(self, mazes):
        """Tile ``mazes`` into a roughly-square grid figure via ``MazeVisualizer.draw``."""
        import math

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        visualizer = MazeVisualizer()
        cols = math.ceil(math.sqrt(len(mazes)))
        rows = math.ceil(len(mazes) / cols)
        fig, _ = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        for ax, maze in zip(fig.axes, mazes):
            visualizer.draw(maze, ax)
        for ax in fig.axes[len(mazes):]:  # blank any unused cells
            ax.set_axis_off()
        fig.tight_layout()
        return fig




class _PruningCallback(pl.Callback):
    """Report ``val_loss`` to an Optuna trial and prune unpromising runs.

    A lightweight stand-in for ``optuna.integration.PyTorchLightningPruningCallback``
    so we don't depend on the separately-versioned ``optuna-integration`` package.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
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
