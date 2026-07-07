"""Typer CLI for generating example mazes and saving visualizations."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import typer

from .maze import BFSMazeGenerator, CarveStrategy, MazeChecker, MazeVisualizer
from.rnn import app as rnn_app

app = typer.Typer(help="Generate and visualise BFS mazes.", add_completion=False)

app.add_typer(rnn_app.app, name="rnn", help="Train an RNN to generate mazes.")



@app.command()
def generate(
    count: int = typer.Option(5, "--count", "-n", min=1, help="Number of mazes to generate."),
    width: int = typer.Option(11, "--width", "-w", help="Maze width (odd number >= 3)."),
    height: int = typer.Option(11, "--height", "-h", help="Maze height (odd number >= 3)."),
    output_dir: Path = typer.Option(Path("mazes"), "--output-dir", "-o", help="Where to save images."),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Seed for reproducible output."),
    strategy: CarveStrategy = typer.Option(
        CarveStrategy.random, "--strategy", help="Carving strategy: random (Prim's) or bfs."
    ),
) -> None:
    """Generate ``count`` example mazes and save a PNG visualization of each."""
    try:
        generator = BFSMazeGenerator(width=width, height=height, strategy=strategy)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    rng = random.Random(seed)
    checker = MazeChecker()
    visualizer = MazeVisualizer()
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        maze = generator.generate(rng)
        if not checker.check(maze):
            # Should never happen: BFS carving always produces a proper maze.
            typer.secho(f"maze {i:03d} failed the proper-maze check!", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        path = visualizer.visualize(maze, output_dir / f"maze_{i:03d}.png")
        typer.echo(f"[{i + 1}/{count}] proper maze saved to {path}")

    typer.secho(f"Done. {count} maze(s) written to {output_dir}/", fg=typer.colors.GREEN)




if __name__ == "__main__":
    app()
