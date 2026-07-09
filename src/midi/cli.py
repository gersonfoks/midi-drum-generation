"""Typer CLI for the MIDI drum tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .data.groove_dataset import DEFAULT_ROOT, ROLAND_TO_GM, GrooveDataset
from .rnn import app as rnn_app
from .synth import FluidSynthSynthesizer

app = typer.Typer(help="MIDI CLI tool for managing MIDI devices and files.")

app.add_typer(rnn_app.app, name="rnn", help="Train an RNN to generate grooves.")


@app.callback()
def main() -> None:
    """MIDI drum tooling."""


@app.command()
def generate_wav(
    input_dir: Path = typer.Option(
        DEFAULT_ROOT, "--input-dir", "-i", help="Root of the raw groove dataset."
    ),
    output_dir: Path = typer.Option(
        Path("data/processed/groove"), "--output-dir", "-o", help="Where to write WAVs."
    ),
    split: Optional[str] = typer.Option(
        None, "--split", "-s", help="Only render this split (train/validation/test)."
    ),
    soundfont: Optional[Path] = typer.Option(
        None,
        "--soundfont",
        help="SoundFont path. Default: the managed GeneralUser GS drum kit "
        "(auto-downloaded to assets/ on first use).",
    ),
    sample_rate: int = typer.Option(44100, "--sample-rate", help="Output sample rate (Hz)."),
    gm_remap: bool = typer.Option(
        True,
        "--gm-remap/--no-gm-remap",
        help="Remap Roland drum pitches to General MIDI so the SoundFont's kit sounds right.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite/--no-overwrite", help="Re-render files that already exist."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", min=1, help="Only render the first N files (for testing)."
    ),
) -> None:
    """Render the Groove MIDI dataset to WAV files, mirroring the raw layout."""
    dataset = GrooveDataset(input_dir)
    datapoints = dataset.split(split) if split else list(dataset)  # type: ignore[arg-type]
    if not datapoints:
        typer.secho("No datapoints selected.", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)
    if limit is not None:
        datapoints = datapoints[:limit]

    synth = FluidSynthSynthesizer(soundfont=soundfont, sample_rate=sample_rate)

    total = len(datapoints)
    written = skipped = failed = 0
    for i, dp in enumerate(datapoints, start=1):
        rel = dp.midi_path.relative_to(input_dir).with_suffix(".wav")
        out_path = output_dir / rel
        if out_path.exists() and not overwrite:
            typer.echo(f"[{i}/{total}] skip (exists) {rel}")
            skipped += 1
            continue
        try:
            midi = dp.midi.remap_pitches(ROLAND_TO_GM) if gm_remap else dp.midi
            synth.synthesize_to_file(midi, out_path)
            typer.echo(f"[{i}/{total}] wrote {rel}")
            written += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            typer.secho(f"[{i}/{total}] FAILED {rel}: {exc}", fg=typer.colors.RED, err=True)
            failed += 1

    color = typer.colors.RED if failed else typer.colors.GREEN
    typer.secho(
        f"Done. {written} written, {skipped} skipped, {failed} failed -> {output_dir}/",
        fg=color,
    )
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
