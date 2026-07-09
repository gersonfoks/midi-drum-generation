"""A training callback that renders model samples to audio you can listen to.

Each run (every ``every_n_epochs`` validation epochs) the model generates a fixed
set of *pre-selected* grooves — the same ``(genre, fill)`` prompts every time, so
you can hear the same conditions improve over training — covering both beats and
fills. Each groove is decoded back to a drum MIDI, synthesised with FluidSynth,
and:

* logged to TensorBoard's **Audio** tab (``add_audio``), and
* optionally written as a ``.wav`` file under the run's log directory.

Synthesis needs the ``fluidsynth`` binary; if it is missing the callback warns
once and becomes a no-op rather than breaking training.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import lightning as pl
import torch

from ..rnn.dataset import tokens_to_midi
from ..synth import FluidSynthSynthesizer

#: Default prompts: a handful of genres, each as a beat and a fill. ``(genre_id,
#: fill_id, label)`` where ids index the model's conditioning embeddings (genre =
#: ``BeatType`` order, fill = ``{beat: 0, fill: 1}``).
DEFAULT_PROMPTS: list[tuple[int, int, str]] = [
    (16, 0, "rock_beat"),   # rock
    (16, 1, "rock_fill"),
    (5, 0, "funk_beat"),    # funk
    (5, 1, "funk_fill"),
    (9, 0, "jazz_beat"),    # jazz
    (9, 1, "jazz_fill"),
    (8, 0, "hiphop_beat"),  # hiphop
    (8, 1, "hiphop_fill"),
]


class GenerateWavCallback(pl.Callback):
    """Generate, synthesise, and log audio samples of the model's grooves.

    Uses FluidSynth to turn generated MIDI into audio, logging up to
    ``len(prompts)`` grooves (both beats and fills) to TensorBoard and, when
    ``out_dir`` is set, to ``.wav`` files.
    """

    def __init__(
        self,
        tokenizer,
        bos_token_id: int,
        prompts: Optional[list[tuple[int, int, str]]] = None,
        gen_length: int = 512,
        temperature: float = 1.0,
        sample_rate: int = 44100,
        every_n_epochs: int = 5,
        out_dir: Optional[str | Path] = None,
        tag_prefix: str = "samples",
    ) -> None:
        self.tokenizer = tokenizer
        self.bos_token_id = bos_token_id
        self.prompts = prompts if prompts is not None else DEFAULT_PROMPTS
        self.gen_length = gen_length
        self.temperature = temperature
        self.sample_rate = sample_rate
        self.every_n_epochs = max(1, every_n_epochs)
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.tag_prefix = tag_prefix
        self._synth: Optional[FluidSynthSynthesizer] = None
        self._synth_unavailable = False

    def _synthesizer(self) -> Optional[FluidSynthSynthesizer]:
        """Lazily build the synthesizer; warn and disable if fluidsynth is absent."""
        if self._synth is not None or self._synth_unavailable:
            return self._synth
        try:
            self._synth = FluidSynthSynthesizer(sample_rate=self.sample_rate)
        except RuntimeError as exc:  # no fluidsynth binary on PATH
            self._synth_unavailable = True
            warnings.warn(
                f"GenerateWavCallback disabled: {exc}. Audio samples will not be logged."
            )
        return self._synth

    def _should_run(self, trainer: pl.Trainer) -> bool:
        if trainer.sanity_checking:
            return False
        epoch = trainer.current_epoch
        is_last = epoch == trainer.max_epochs - 1 if trainer.max_epochs else False
        return epoch % self.every_n_epochs == 0 or is_last

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self._should_run(trainer):
            return
        synth = self._synthesizer()
        if synth is None:
            return

        genre_ids = torch.tensor([p[0] for p in self.prompts], dtype=torch.long)
        fill_ids = torch.tensor([p[1] for p in self.prompts], dtype=torch.long)
        batch = pl_module.generate_batch(  # type: ignore[attr-defined]
            self.gen_length,
            genre_ids,
            fill_ids,
            self.bos_token_id,
            temperature=self.temperature,
        )

        experiment = getattr(trainer.logger, "experiment", None)
        epoch = trainer.current_epoch
        for (_, _, label), ids in zip(self.prompts, batch.tolist()):
            midi = tokens_to_midi(self.tokenizer, ids)
            if midi is None:
                continue  # model emitted nothing decodable for this prompt
            try:
                audio = synth.render(midi)  # (channels, samples), float32 in [-1, 1]
            except Exception as exc:  # noqa: BLE001 - one bad sample shouldn't stop training
                warnings.warn(f"GenerateWavCallback: failed to synthesise {label!r}: {exc}")
                continue
            mono = audio.mean(axis=0)  # downmix to mono for add_audio

            if experiment is not None and hasattr(experiment, "add_audio"):
                experiment.add_audio(
                    f"{self.tag_prefix}/{label}",
                    torch.from_numpy(mono),
                    global_step=epoch,
                    sample_rate=self.sample_rate,
                )
            if self.out_dir is not None:
                path = self.out_dir / f"epoch={epoch:03d}_{label}.wav"
                synth.synthesize_to_file(midi, path)
