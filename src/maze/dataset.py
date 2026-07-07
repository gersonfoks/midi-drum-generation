"""Turn mazes into fixed-length integer token sequences for the RNN.

Every maze produced by :class:`~src.maze.maze.BFSMazeGenerator` is a fixed
``height x width`` grid, so it flattens to a fixed-length sequence of
:class:`~src.maze.maze.MazeCell` values (``0``-``3``). We prepend a dedicated
*begin-of-sequence* token so the model has something to condition its first real
prediction on; because ``MazeCell.wall`` is already ``0`` the BOS token takes the
next free id (``4``), giving a vocabulary of :data:`VOCAB_SIZE` tokens.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Dataset

from .maze import BFSMazeGenerator, Maze, MazeCell

#: The begin-of-sequence token id. It sits just past the four ``MazeCell``
#: values (``wall=0 .. end=3``) so it never collides with a real cell.
BOS_TOKEN = 4

#: Number of distinct tokens: the four cell values plus :data:`BOS_TOKEN`.
VOCAB_SIZE = 5

#: Maps an integer token back to the cell it encodes (BOS has no cell).
_TOKEN_TO_CELL: dict[int, MazeCell] = {cell.value: cell for cell in MazeCell}


def maze_to_tokens(maze: Maze) -> list[int]:
    """Flatten ``maze`` (row-major) into ``[BOS, cell, cell, ...]`` token ids."""
    tokens = [BOS_TOKEN]
    for row in maze.grid:
        tokens.extend(cell.value for cell in row)
    return tokens


def tokens_to_maze(tokens: list[int], height: int, width: int) -> Maze:
    """Rebuild a :class:`Maze` from a token sequence (inverse of
    :func:`maze_to_tokens`).

    A leading :data:`BOS_TOKEN` is stripped only when it is *extra* (i.e. the
    sequence is one longer than the grid), so a sampled sequence whose first cell
    token merely happens to equal the BOS id is left intact. The remaining
    ``height * width`` tokens are mapped back to :class:`MazeCell` values; unknown
    ids (which a partly-trained model may emit) fall back to ``MazeCell.wall`` so
    decoding never raises.
    """
    expected = height * width
    if len(tokens) == expected + 1 and tokens and tokens[0] == BOS_TOKEN:
        cells = tokens[1:]
    else:
        cells = tokens
    if len(cells) < expected:
        raise ValueError(
            f"need {expected} cell tokens for a {height}x{width} maze, got {len(cells)}"
        )
    grid = [
        [_TOKEN_TO_CELL.get(cells[i * width + j], MazeCell.wall) for j in range(width)]
        for i in range(height)
    ]
    return Maze(grid=grid)


class MazeSequenceDataset(Dataset):
    """A dataset of mazes tokenised into fixed-length ``torch.long`` sequences.

    Mazes are generated up-front from a seeded :class:`random.Random` so the
    dataset is fully reproducible for a given ``seed``. Each item is a 1-D tensor
    of length ``height * width + 1`` (the extra element is :data:`BOS_TOKEN`).
    """

    def __init__(self, generator: BFSMazeGenerator, num_samples: int, seed: int = 0):
        rng = random.Random(seed)
        self.samples: list[torch.Tensor] = [
            torch.tensor(maze_to_tokens(generator.generate(rng)), dtype=torch.long)
            for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.samples[index]


def build_dataloaders(
    *,
    maze_size: int,
    carve_strategy,
    num_train_samples: int,
    batch_size: int,
    seed: int = 0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build reproducible train/val/test dataloaders.

    Validation and test sets are each 10% of the training size. The three splits
    use distinct seeds (derived from ``seed``) so they contain different mazes.
    """
    generator = BFSMazeGenerator(width=maze_size, height=maze_size, strategy=carve_strategy)
    num_val = max(1, int(num_train_samples * 0.1))
    num_test = max(1, int(num_train_samples * 0.1))

    train = MazeSequenceDataset(generator, num_train_samples, seed=seed)
    val = MazeSequenceDataset(generator, num_val, seed=seed + 1)
    test = MazeSequenceDataset(generator, num_test, seed=seed + 2)

    return (
        DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True),
        DataLoader(val, batch_size=batch_size, num_workers=num_workers, shuffle=False),
        DataLoader(test, batch_size=batch_size, num_workers=num_workers, shuffle=False),
    )
