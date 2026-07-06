"""Grid mazes for use as a toy problem for generative models.

A maze is represented as a 2D grid of :class:`MazeCell` values. Cells at odd
coordinates ``(i, j)`` (with ``i`` and ``j`` both odd) are *path* cells; the
cells between two path cells are *bridges*. A bridge is a wall until it is
carved open, at which point the two path cells it separates become connected.

Because the maze is carved as a spanning tree over the path cells, a correctly
generated maze is *proper*: every path cell is reachable from the start and the
maze contains no loops. :class:`MazeChecker` verifies both properties.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# The four grid-neighbours of a path cell live two steps away; the bridge that
# separates them is the cell halfway between (``step // 2``).
_STEPS: tuple[tuple[int, int], ...] = ((-2, 0), (2, 0), (0, -2), (0, 2))


class MazeCell(Enum):
    """A single cell of a maze grid.

    Values double as the integer encoding used when rasterising a maze for a
    model (walls are ``0``, open path is ``1``).
    """

    wall = 0
    path = 1
    start = 2
    end = 3


@dataclass
class Maze:
    """A maze as a rectangular grid of :class:`MazeCell` values."""

    grid: list[list[MazeCell]]

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.height and 0 <= y < self.width

    def path_cells(self) -> list[tuple[int, int]]:
        """All path cells (the cells at odd coordinates), including start/end."""
        return [
            (i, j)
            for i in range(1, self.height, 2)
            for j in range(1, self.width, 2)
        ]

    def get_neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """Return the path cells connected to ``cell`` through a carved bridge.

        Two path cells are neighbours only if the bridge between them has been
        carved open (is not a wall). This is the maze's connectivity graph, used
        both to solve the maze and to check that it is proper.
        """
        x, y = cell
        neighbors = []
        for dx, dy in _STEPS:
            nx, ny = x + dx, y + dy
            if self.in_bounds((nx, ny)) and self.grid[x + dx // 2][y + dy // 2] != MazeCell.wall:
                neighbors.append((nx, ny))
        return neighbors


class CarveStrategy(str, Enum):
    """How :class:`BFSMazeGenerator` grows the spanning tree.

    ``random`` (default) pops a random frontier edge each step (randomised
    Prim's): winding corridors and more varied mazes. ``bfs`` grows the tree in
    breadth-first order: bushier mazes with short corridors radiating from the
    start.
    """

    random = "random"
    bfs = "bfs"


class BFSMazeGenerator:
    """Generate mazes by carving a spanning tree over the path cells.

    Both strategies (see :class:`CarveStrategy`) discover every path cell exactly
    once and carve exactly one bridge per discovery, so the result is always a
    spanning tree: fully connected and loop-free.

    Width and height must be odd numbers ``>= 3``: path cells sit at odd
    coordinates and the outer ring stays walls, so the start ``(1, 1)`` and end
    ``(height - 2, width - 2)`` land on path cells.
    """

    def __init__(
        self,
        width: int,
        height: int,
        strategy: CarveStrategy = CarveStrategy.random,
    ):
        if width < 3 or height < 3 or width % 2 == 0 or height % 2 == 0:
            raise ValueError(
                f"width and height must be odd numbers >= 3, got {width}x{height}"
            )
        self.width = width
        self.height = height
        self.strategy = strategy

    def generate(self, rng: random.Random | None = None) -> Maze:
        """Generate a proper maze. Pass a seeded ``rng`` for reproducibility."""
        rng = rng or random.Random()
        grid = [[MazeCell.wall for _ in range(self.width)] for _ in range(self.height)]
        for i in range(1, self.height, 2):
            for j in range(1, self.width, 2):
                grid[i][j] = MazeCell.path
        maze = Maze(grid=grid)

        start = (1, 1)
        end = (self.height - 2, self.width - 2)
        self._carve(maze, start, rng)

        grid[start[0]][start[1]] = MazeCell.start
        grid[end[0]][end[1]] = MazeCell.end
        return maze

    def _carve(self, maze: Maze, start: tuple[int, int], rng: random.Random) -> None:
        """Carve bridges into a spanning tree using the configured strategy."""
        if self.strategy is CarveStrategy.random:
            self._carve_random(maze, start, rng)
        else:
            self._carve_bfs(maze, start, rng)

    def _carve_random(self, maze: Maze, start: tuple[int, int], rng: random.Random) -> None:
        """Randomised Prim's: repeatedly carve a random edge on the frontier.

        The frontier is the set of ``(visited_cell, unvisited_cell)`` edges. Each
        step pops a random edge; if its far cell is still unvisited, that single
        bridge is carved and the cell's own edges join the frontier. Picking a
        random edge (rather than a random neighbour of one cell) yields winding,
        varied mazes.
        """
        visited = {start}
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = [
            (start, nb) for nb in self._candidate_neighbors(maze, start)
        ]
        while frontier:
            index = rng.randrange(len(frontier))
            frontier[index], frontier[-1] = frontier[-1], frontier[index]
            from_cell, to_cell = frontier.pop()
            if to_cell in visited:
                continue
            visited.add(to_cell)
            self._carve_bridge(maze, from_cell, to_cell)
            frontier.extend(
                (to_cell, nb)
                for nb in self._candidate_neighbors(maze, to_cell)
                if nb not in visited
            )

    def _carve_bfs(self, maze: Maze, start: tuple[int, int], rng: random.Random) -> None:
        """Breadth-first carving: expand the tree level by level from the start.

        Each cell's neighbours are visited in random order so different seeds
        yield different mazes.
        """
        visited = {start}
        frontier: deque[tuple[int, int]] = deque([start])
        while frontier:
            cell = frontier.popleft()
            neighbors = self._candidate_neighbors(maze, cell)
            rng.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self._carve_bridge(maze, cell, neighbor)
                    frontier.append(neighbor)

    @staticmethod
    def _candidate_neighbors(maze: Maze, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """The in-bounds path cells two steps away from ``cell`` (ignoring walls)."""
        x, y = cell
        return [
            (x + dx, y + dy)
            for dx, dy in _STEPS
            if maze.in_bounds((x + dx, y + dy))
        ]

    @staticmethod
    def _carve_bridge(maze: Maze, a: tuple[int, int], b: tuple[int, int]) -> None:
        """Open the bridge cell between adjacent path cells ``a`` and ``b``."""
        maze.grid[(a[0] + b[0]) // 2][(a[1] + b[1]) // 2] = MazeCell.path


class MazeChecker:
    """Verify that a maze is *proper*: fully reachable and loop-free."""

    def check(self, maze: Maze) -> bool:
        return self.check_reachability(maze) and self.check_no_loops(maze)

    def check_reachability(self, maze: Maze) -> bool:
        """Every path cell is reachable from the start cell ``(1, 1)``."""
        cells = maze.path_cells()
        if not cells:
            return False
        start = cells[0]
        visited = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in maze.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return len(visited) == len(cells)

    def check_no_loops(self, maze: Maze) -> bool:
        """The connectivity graph contains no cycles (union-find over edges)."""
        cells = maze.path_cells()
        parent = {cell: cell for cell in cells}

        def find(cell: tuple[int, int]) -> tuple[int, int]:
            while parent[cell] != cell:
                parent[cell] = parent[parent[cell]]
                cell = parent[cell]
            return cell

        for cell in cells:
            for neighbor in maze.get_neighbors(cell):
                if neighbor <= cell:  # visit each undirected edge only once
                    continue
                root_a, root_b = find(cell), find(neighbor)
                if root_a == root_b:  # the edge closes a cycle
                    return False
                parent[root_a] = root_b
        return True


class MazeVisualizer:
    """Render mazes to image files for inspection."""

    #: Cell -> fill colour. Walls dark, path light, start green, end red.
    _COLORS: dict[MazeCell, str] = {
        MazeCell.wall: "#1a1a1a",
        MazeCell.path: "#f5f5f5",
        MazeCell.start: "#2ecc71",
        MazeCell.end: "#e74c3c",
    }

    def __init__(self, cell_size: float = 0.3):
        #: Size of one cell in inches; controls the output resolution.
        self.cell_size = cell_size

    def visualize(self, maze: Maze, save_path: str | Path) -> Path:
        """Render ``maze`` to ``save_path`` (PNG) and return the written path."""
        import matplotlib

        matplotlib.use("Agg")  # headless backend, no display required
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        order = [MazeCell.wall, MazeCell.path, MazeCell.start, MazeCell.end]
        index = {cell: i for i, cell in enumerate(order)}
        raster = [[index[cell] for cell in row] for row in maze.grid]
        cmap = ListedColormap([self._COLORS[cell] for cell in order])

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(
            figsize=(maze.width * self.cell_size, maze.height * self.cell_size)
        )
        ax.imshow(raster, cmap=cmap, vmin=0, vmax=len(order) - 1)
        ax.set_axis_off()
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=100)
        plt.close(fig)
        return save_path
