from src.maze.cli import app as maze_app
import typer

app = typer.Typer(help="Maze generation toolkit.", add_completion=False)

app.add_typer(maze_app, name="maze", help="Generate and visualise BFS mazes.")



if __name__ == "__main__":
    app()
