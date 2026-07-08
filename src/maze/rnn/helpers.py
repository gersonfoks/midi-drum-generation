from src.maze.dataset import tokens_to_maze
from src.maze.maze import MazeChecker
from src.maze.rnn.lightning_model import LightningRnnModel


def sample_maze(model: LightningRnnModel, maze_size, temperature: float = 1.0):
    """Decode a single maze from ``model`` and report whether it is a *proper* maze."""
    sequence_length = maze_size * maze_size
    tokens = model.generate(sequence_length, temperature=temperature)
    maze = tokens_to_maze(tokens, maze_size, maze_size)
    return maze, MazeChecker().check(maze)


def sample_mazes(model: LightningRnnModel, maze_size, count: int, temperature: float = 1.0):
    """Sample ``count`` mazes in parallel and return them with proper/valid flags.

    Generation is batched (one autoregressive pass for all ``count`` sequences via
    :meth:`LightningRnnModel.generate_batch`); the cheap per-maze decode + validity
    check then runs on CPU. ``temperature=1.0`` samples from the real learned
    distribution, so the grids differ.
    """
    sequence_length = maze_size * maze_size
    batch = model.generate_batch(sequence_length, count, temperature=temperature)
    checker = MazeChecker()
    mazes, valid = [], []
    for tokens in batch.tolist():
        maze = tokens_to_maze(tokens, maze_size, maze_size)
        mazes.append(maze)
        valid.append(checker.check(maze))
    return mazes, valid
