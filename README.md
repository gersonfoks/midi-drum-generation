# midi-drum-generator

## Mazes

A small maze toolkit (`src/maze`) intended as a toy problem for generative
deep-learning models. Mazes are grids of `MazeCell` values; path cells sit at
odd coordinates and the cells between them are bridges that get carved open.

- **`BFSMazeGenerator`** carves a spanning tree over the path cells. Two
  strategies (`CarveStrategy`): `random` (default) pops a random frontier edge
  each step (randomised Prim's) for winding, varied mazes; `bfs` grows the tree
  breadth-first for bushier mazes with short corridors radiating from the start.
  Width and height must be odd numbers `>= 3`. Pass a seeded `random.Random` for
  reproducible output.
- **`MazeChecker`** verifies a maze is *proper*: `check_reachability` (every
  path cell reachable from the start) and `check_no_loops` (the connectivity
  graph is acyclic). Generated mazes always pass.
- **`MazeVisualizer`** renders a maze to a PNG (walls dark, path light, start
  green, end red).

### CLI

Generate `n` example mazes and save a visualization of each:

```bash
uv run python main.py generate -n 10 --width 21 --height 21 --seed 0 -o mazes
```

Options: `--count/-n`, `--width/-w`, `--height/-h`, `--output-dir/-o`,
`--seed/-s`, `--strategy` (`random` or `bfs`). See
`uv run python main.py generate --help`.

### Tests

```bash
uv run pytest
```
