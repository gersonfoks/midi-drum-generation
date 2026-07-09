"""Render :class:`~src.midi.data.midi.Midi` objects to WAV audio.

MIDI carries no audio of its own, so turning a groove into a ``.wav`` requires a
synthesiser and a SoundFont. :class:`FluidSynthSynthesizer` renders through the
system ``fluidsynth`` engine, which produces markedly more realistic drums than
lightweight in-process synths. It needs a ``fluidsynth`` binary on ``PATH`` and a
SoundFont; with no SoundFont supplied it uses the managed default (GeneralUser GS,
auto-downloaded on first use — see :func:`src.midi.soundfonts.ensure_soundfont`).

fluidsynth's file renderer appends a long stretch of silence after the song ends,
and its reverb adds a smooth multi-second tail on top. Rendered audio is therefore
trimmed to the MIDI's own content length (from :attr:`Midi.content_end`, the last
note-off) plus a fixed decay tail — deterministic, and immune to the reverb floor
that defeats amplitude-based silence detection.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from .data.midi import Midi


class FluidSynthSynthesizer:
    """Render :class:`Midi` to WAV via the system ``fluidsynth`` engine."""

    def __init__(
        self,
        soundfont: Optional[str | Path] = None,
        sample_rate: int = 44100,
        tail_seconds: float = 2.0,
    ) -> None:
        if shutil.which("fluidsynth") is None:
            raise RuntimeError(
                "The 'fluidsynth' binary was not found on PATH. Install it "
                "(e.g. 'apt install fluidsynth' or 'brew install fluid-synth')."
            )
        self.sample_rate = sample_rate
        # Seconds of audio to keep past the MIDI's last event, for the note/reverb
        # decay to ring out.
        self.tail_seconds = tail_seconds
        if soundfont is None:
            from .soundfonts import ensure_soundfont

            self.soundfont = ensure_soundfont()
        else:
            self.soundfont = Path(soundfont)
            if not self.soundfont.is_file():
                raise FileNotFoundError(f"SoundFont not found: {self.soundfont}")

    def synthesize_to_file(self, midi: Midi, out_path: str | Path) -> Path:
        """Render ``midi`` to ``out_path`` as a WAV file.

        Returns the path written. Parent directories are created as needed.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames, channels = self._render_frames(midi)
        frames = self._trim_to_content(frames, channels, midi.content_end)
        with wave.open(str(out_path), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)  # int16
            wav.setframerate(self.sample_rate)
            wav.writeframes(frames.tobytes())
        return out_path

    def render(self, midi: Midi) -> np.ndarray:
        """Render ``midi`` to a float32 array shaped ``(channels, samples)``."""
        frames, channels = self._render_frames(midi)
        frames = self._trim_to_content(frames, channels, midi.content_end)
        return frames.reshape(-1, channels).T.astype(np.float32) / 32768.0

    def _render_frames(self, midi: Midi) -> tuple[np.ndarray, int]:
        """Run fluidsynth and return its raw int16 samples ``(interleaved,)`` + channels."""
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as mtmp:
            mid_path = Path(mtmp.name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wtmp:
            wav_path = Path(wtmp.name)
        try:
            midi.to_file(mid_path)
            subprocess.run(
                [
                    "fluidsynth",
                    "-ni",
                    "-r",
                    str(self.sample_rate),
                    "-F",
                    str(wav_path),
                    str(self.soundfont),
                    str(mid_path),
                ],
                check=True,
                capture_output=True,
            )
            with wave.open(str(wav_path), "rb") as wav:
                channels = wav.getnchannels()
                frames = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            return frames, channels
        finally:
            mid_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)

    def _trim_to_content(
        self, frames: np.ndarray, channels: int, content_seconds: float
    ) -> np.ndarray:
        """Cut fluidsynth's trailing silence/reverb to the MIDI content + tail.

        ``frames`` is interleaved int16; the returned array is also interleaved.
        """
        stereo = frames.reshape(-1, channels)
        keep = int((content_seconds + self.tail_seconds) * self.sample_rate)
        keep = max(0, min(keep, stereo.shape[0]))
        return stereo[:keep].reshape(-1)
