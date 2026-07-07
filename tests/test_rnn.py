import random

import torch

from src.maze.dataset import (
    BOS_TOKEN,
    VOCAB_SIZE,
    MazeSequenceDataset,
    build_dataloaders,
    maze_to_tokens,
    tokens_to_maze,
)
from src.maze.maze import BFSMazeGenerator, CarveStrategy, MazeChecker, MazeVisualizer
from src.maze.rnn.app import _smoke_test
from src.maze.rnn.model import RNNModel
from src.maze.rnn.train import TrainConfig, build_model, sample_mazes


# --- encoding -------------------------------------------------------------


def test_token_roundtrip_reproduces_grid():
    generator = BFSMazeGenerator(width=11, height=11)
    maze = generator.generate(random.Random(0))

    tokens = maze_to_tokens(maze)
    assert tokens[0] == BOS_TOKEN
    assert len(tokens) == maze.height * maze.width + 1

    rebuilt = tokens_to_maze(tokens, maze.height, maze.width)
    assert rebuilt.grid == maze.grid


def test_tokens_to_maze_strips_leading_bos_only():
    # A raw (BOS-less) token list should also decode to the right shape.
    tokens = [1] * (5 * 5)
    maze = tokens_to_maze(tokens, 5, 5)
    assert maze.height == 5 and maze.width == 5


# --- dataset --------------------------------------------------------------


def test_dataset_item_shape_and_dtype():
    generator = BFSMazeGenerator(width=7, height=7)
    dataset = MazeSequenceDataset(generator, num_samples=4, seed=0)

    assert len(dataset) == 4
    item = dataset[0]
    assert item.dtype == torch.long
    assert item.shape == (7 * 7 + 1,)
    assert item[0].item() == BOS_TOKEN


def test_dataset_is_reproducible_for_a_seed():
    generator = BFSMazeGenerator(width=7, height=7)
    a = MazeSequenceDataset(generator, num_samples=3, seed=42)
    b = MazeSequenceDataset(generator, num_samples=3, seed=42)
    assert all(torch.equal(x, y) for x, y in zip(a.samples, b.samples))


def test_build_dataloaders_batches_fixed_length_sequences():
    train, val, test = build_dataloaders(
        maze_size=5,
        carve_strategy=CarveStrategy.random,
        num_train_samples=20,
        batch_size=4,
        seed=0,
    )
    batch = next(iter(train))
    assert batch.shape == (4, 5 * 5 + 1)
    assert len(val.dataset) == 2 and len(test.dataset) == 2  # 10% each


# --- model ----------------------------------------------------------------


def test_forward_returns_per_timestep_logits():
    model = RNNModel(vocab_size=VOCAB_SIZE, embedding_dim=8, hidden_size=16, n_layers=1)
    x = torch.zeros(3, 10, dtype=torch.long)
    logits, _ = model(x)
    assert logits.shape == (3, 10, VOCAB_SIZE)


def test_shared_step_gives_finite_loss():
    config = TrainConfig(maze_size=5, hidden_size=16, embedding_dim=8, n_layers=1)
    model = build_model(config)
    batch = torch.randint(0, VOCAB_SIZE, (2, config.sequence_length + 1))
    loss, accuracy = model.shared_step(batch)
    assert torch.isfinite(loss)
    assert 0.0 <= float(accuracy) <= 1.0


# --- visualization + validity --------------------------------------------


def test_visualizer_draw_paints_on_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maze = BFSMazeGenerator(width=7, height=7).generate(random.Random(0))
    fig, ax = plt.subplots()
    MazeVisualizer().draw(maze, ax)  # should not raise
    assert ax.images, "draw did not add an image to the axes"
    plt.close(fig)


def test_generated_mazes_get_a_validity_flag():
    config = TrainConfig(maze_size=5, hidden_size=16, embedding_dim=8, n_layers=1)
    model = build_model(config)
    mazes, valid = sample_mazes(model, config.maze_size, count=8)
    assert len(mazes) == len(valid) == 8
    assert all(isinstance(v, bool) for v in valid)
    # An untrained model is unlikely to emit proper mazes, but flags must be
    # consistent with MazeChecker on each grid.
    checker = MazeChecker()
    assert valid == [checker.check(m) for m in mazes]


def test_generate_batch_shape_and_diversity():
    config = TrainConfig(maze_size=5, hidden_size=16, embedding_dim=8, n_layers=1)
    model = build_model(config)
    length, count = config.sequence_length, 16
    batch = model.generate_batch(length, count, temperature=1.0)
    assert batch.shape == (count, length)
    # Sampling (temperature 1.0), not greedy: the rows should not all be identical.
    distinct = {tuple(row) for row in batch.tolist()}
    assert len(distinct) >= 2, "temperature sampling produced identical sequences"


# --- end to end -----------------------------------------------------------


def test_smoke_test_runs():
    _smoke_test()
