"""Tests for the Groove dataset loader, MIDI wrapper, and WAV synthesis."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.midi.data.groove_dataset import (
    ROLAND_TO_GM,
    BeatType,
    GrooveDataPoint,
    GrooveDataset,
)
from src.midi.data.midi import Midi
from src.midi.soundfonts import DEFAULT_SOUNDFONT, ensure_soundfont
from src.midi.synth import FluidSynthSynthesizer

ROOT = Path("data/raw/groove")

pytestmark = pytest.mark.skipif(
    not (ROOT / "info.csv").is_file(),
    reason="raw groove dataset not present under data/raw/groove",
)


@pytest.fixture(scope="module")
def dataset() -> GrooveDataset:
    return GrooveDataset(ROOT)


# --- dataset loading ------------------------------------------------------


def test_loads_all_rows(dataset: GrooveDataset) -> None:
    assert len(dataset) == 1150


def test_splits_partition_the_dataset(dataset: GrooveDataset) -> None:
    counts = {s: len(dataset.split(s)) for s in ("train", "validation", "test")}
    assert sum(counts.values()) == len(dataset)
    assert all(v > 0 for v in counts.values())


def test_datapoint_fields_parsed(dataset: GrooveDataset) -> None:
    dp = dataset[0]
    assert isinstance(dp, GrooveDataPoint)
    assert dp.primary_beat_type in BeatType
    assert dp.groove_type in ("beat", "fill")
    assert dp.split in ("train", "validation", "test")
    assert dp.bpm > 0
    assert dp.duration > 0
    assert dp.midi_path.is_file()


def test_style_splits_into_primary_and_secondary(dataset: GrooveDataset) -> None:
    with_secondary = next(dp for dp in dataset if "/" in dp.style)
    primary, secondary = with_secondary.style.split("/", 1)
    assert with_secondary.primary_beat_type.value == primary
    assert with_secondary.secondary_beat_type == secondary


# --- MIDI wrapper ---------------------------------------------------------


def test_midi_loads_and_matches_reported_duration(dataset: GrooveDataset) -> None:
    dp = dataset[0]
    midi = dp.midi
    assert isinstance(midi, Midi)
    assert len(midi.score.tracks) >= 1
    assert midi.duration == pytest.approx(dp.duration, abs=0.5)


def test_midi_is_cached(dataset: GrooveDataset) -> None:
    dp = dataset[0]
    assert dp.midi is dp.midi


def test_midi_roundtrips_to_file(dataset: GrooveDataset, tmp_path: Path) -> None:
    out = tmp_path / "roundtrip.mid"
    dataset[0].midi.to_file(out)
    assert out.is_file()
    assert Midi.from_file(out).duration > 0


def _pitches(midi: Midi) -> set[int]:
    return {n.pitch for t in midi.score.tracks for n in t.notes}


def test_gm_remap_removes_non_gm_pitches(dataset: GrooveDataset) -> None:
    # A groove that actually uses the sub-GM hi-hat "edge" pitches (22/26).
    dp = max(
        dataset,
        key=lambda d: len(_pitches(d.midi) & {22, 26, 58}),
    )
    assert _pitches(dp.midi) & {22, 26, 58}, "fixture groove should exercise the remap"

    remapped = dp.midi.remap_pitches(ROLAND_TO_GM)
    assert not (_pitches(remapped) & {22, 26, 58})
    # Every remapped pitch is a valid GM percussion note, and the original is untouched.
    assert all(35 <= p <= 81 for p in _pitches(remapped))
    assert _pitches(dp.midi) & {22, 26, 58}
    # Same number of hits, just retriggered on GM notes.
    before = sum(len(t.notes) for t in dp.midi.score.tracks)
    after = sum(len(t.notes) for t in remapped.score.tracks)
    assert before == after


# --- synthesis ------------------------------------------------------------


def test_default_soundfont_is_present_and_verified() -> None:
    # ensure_soundfont downloads on first use; here it should already be cached.
    # Skip (rather than trigger a network download) if it isn't present yet.
    if not DEFAULT_SOUNDFONT.path.is_file():
        pytest.skip("default SoundFont not downloaded yet (needs network)")
    path = ensure_soundfont()
    assert path == DEFAULT_SOUNDFONT.path
    assert path.stat().st_size == DEFAULT_SOUNDFONT.size


@pytest.mark.slow
def test_fluidsynth_synthesizer_writes_wav(dataset: GrooveDataset, tmp_path: Path) -> None:
    if shutil.which("fluidsynth") is None:
        pytest.skip("fluidsynth binary not installed")
    if not DEFAULT_SOUNDFONT.path.is_file():
        pytest.skip("default SoundFont not downloaded yet (needs network)")
    import wave

    dp = dataset[0]
    out = tmp_path / "groove.wav"
    synth = FluidSynthSynthesizer(sample_rate=22050)
    result = synth.synthesize_to_file(dp.midi.remap_pitches(ROLAND_TO_GM), out)
    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 1000

    # fluidsynth appends a long silent tail (and reverb rings for ~10s); it must be
    # trimmed to the MIDI content plus the configured decay tail, not left at ~2x.
    with wave.open(str(out), "rb") as wav:
        wav_seconds = wav.getnframes() / wav.getframerate()
    assert wav_seconds <= dp.midi.duration + synth.tail_seconds + 0.05
