from typer import Typer

from src.maze.cli import app as maze_app
from src.midi.cli import app as midi_app


app = Typer(help="CLI tool for managing MIDI devices and files.")

app.add_typer(maze_app, name="maze", help="Maze generation and solving tools.")
app.add_typer(midi_app, name="midi", help="MIDI drum tooling.")


if __name__ == "__main__":
    app()