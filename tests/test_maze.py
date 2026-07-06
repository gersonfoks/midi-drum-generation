import random

import pytest

from src.maze.maze import (
    BFSMazeGenerator,
    CarveStrategy,
    Maze,
    MazeCell,
    MazeChecker,
    MazeVisualizer,
)

SIZES = [(3, 3), (5, 5), (11, 11), (7, 13), (21, 15)]


def _grid(rows: list[str]) -> Maze:
    """Build a maze from a compact string grid: '#'=wall, '.'=path, 'S'=start, 'E'=end."""
    mapping = {
        "#": MazeCell.wall,
        ".": MazeCell.path,
        "S": MazeCell.start,
        "E": MazeCell.end,
    }
    return Maze(grid=[[mapping[c] for c in row] for row in rows])


# --- generator ------------------------------------------------------------


@pytest.mark.parametrize("width,height", SIZES)
def test_generate_has_requested_dimensions(width, height):
    maze = BFSMazeGenerator(width, height).generate(random.Random(0))
    assert maze.height == height
    assert maze.width == width


def test_generate_places_start_and_end():
    maze = BFSMazeGenerator(11, 11).generate(random.Random(0))
    assert maze.grid[1][1] == MazeCell.start
    assert maze.grid[9][9] == MazeCell.end


def test_generate_keeps_outer_ring_walls():
    maze = BFSMazeGenerator(11, 9).generate(random.Random(1))
    for j in range(maze.width):
        assert maze.grid[0][j] == MazeCell.wall
        assert maze.grid[maze.height - 1][j] == MazeCell.wall
    for i in range(maze.height):
        assert maze.grid[i][0] == MazeCell.wall
        assert maze.grid[i][maze.width - 1] == MazeCell.wall


@pytest.mark.parametrize("bad", [(2, 5), (5, 2), (4, 4), (1, 3), (3, 1), (0, 3)])
def test_generate_rejects_invalid_dimensions(bad):
    with pytest.raises(ValueError):
        BFSMazeGenerator(*bad)


def test_generation_is_reproducible_with_seed():
    a = BFSMazeGenerator(15, 15).generate(random.Random(42)).grid
    b = BFSMazeGenerator(15, 15).generate(random.Random(42)).grid
    assert a == b


def test_different_seeds_give_different_mazes():
    a = BFSMazeGenerator(15, 15).generate(random.Random(1)).grid
    b = BFSMazeGenerator(15, 15).generate(random.Random(2)).grid
    assert a != b


# --- checker: generated mazes are always proper ---------------------------


def test_default_strategy_is_random():
    assert BFSMazeGenerator(5, 5).strategy is CarveStrategy.random


@pytest.mark.parametrize("strategy", list(CarveStrategy))
@pytest.mark.parametrize("width,height", SIZES)
def test_generated_mazes_are_proper(width, height, strategy):
    checker = MazeChecker()
    generator = BFSMazeGenerator(width, height, strategy=strategy)
    rng = random.Random(123)
    for _ in range(20):
        maze = generator.generate(rng)
        assert checker.check_reachability(maze)
        assert checker.check_no_loops(maze)
        assert checker.check(maze)


def test_strategies_produce_different_mazes():
    rng_a = random.Random(0)
    rng_b = random.Random(0)
    random_maze = BFSMazeGenerator(21, 21, strategy=CarveStrategy.random).generate(rng_a)
    bfs_maze = BFSMazeGenerator(21, 21, strategy=CarveStrategy.bfs).generate(rng_b)
    assert random_maze.grid != bfs_maze.grid


# --- checker: detects broken mazes ----------------------------------------


def test_checker_detects_unreachable_cell():
    # End cell (3, 3) is fully walled off: its bridges (2,3) and (3,2) are walls.
    maze = _grid(
        [
            "#####",
            "S...#",
            "#.###",
            "#.#E#",
            "#####",
        ]
    )
    checker = MazeChecker()
    assert not checker.check_reachability(maze)
    assert not checker.check(maze)


def test_checker_detects_loop():
    # A 2x2 block of path cells fully connected -> a cycle.
    maze = _grid(
        [
            "#####",
            "#...#",
            "#.#.#",
            "#...#",
            "#####",
        ]
    )
    checker = MazeChecker()
    assert checker.check_reachability(maze)  # every cell reachable...
    assert not checker.check_no_loops(maze)  # ...but there is a loop
    assert not checker.check(maze)


def test_checker_accepts_hand_built_tree():
    maze = _grid(
        [
            "#####",
            "S...#",
            "###.#",
            "#..E#",
            "#####",
        ]
    )
    checker = MazeChecker()
    assert checker.check(maze)


# --- Maze.get_neighbors ----------------------------------------------------


def test_get_neighbors_respects_walls():
    maze = _grid(
        [
            "#####",
            "S...#",
            "###.#",
            "#..E#",
            "#####",
        ]
    )
    # (1,1) connects only right to (1,3): bridge (1,2) is open, below (2,1) is wall.
    assert set(maze.get_neighbors((1, 1))) == {(1, 3)}
    # (1,3) connects left to (1,1) and down to (3,3).
    assert set(maze.get_neighbors((1, 3))) == {(1, 1), (3, 3)}


def test_get_neighbors_none_when_isolated():
    maze = _grid(
        [
            "#####",
            "#.#.#",
            "#####",
            "#.#.#",
            "#####",
        ]
    )
    assert maze.get_neighbors((1, 1)) == []


# --- visualizer ------------------------------------------------------------


def test_visualizer_writes_png(tmp_path):
    maze = BFSMazeGenerator(11, 11).generate(random.Random(0))
    out = tmp_path / "sub" / "maze.png"
    written = MazeVisualizer().visualize(maze, out)
    assert written == out
    assert out.exists()
    assert out.stat().st_size > 0
