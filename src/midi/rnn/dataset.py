"""Turn Groove performances into padded token sequences for the RNN.

Each performance is tokenised with a miditok :class:`REMI` tokenizer, wrapped in
``[BOS, ...ids, EOS]``, and paired with two conditioning labels — the *genre*
(:class:`~src.midi.data.groove_dataset.BeatType`, 18 values) and whether the clip
is a *fill* (2 values). Unlike the fixed-size maze grids, MIDI token sequences
vary in length, so we:

* **drop** any clip whose sequence exceeds ``max_seq_len`` (never truncate), and
* right-pad each batch to its longest sequence in :func:`collate_fn`, using the
  tokenizer's ``PAD`` id so the loss can ignore the padding.

The tokenizer itself (:func:`build_tokenizer`) is a REMI tokenizer with
``PAD``/``BOS``/``EOS`` special tokens; with ``use_bpe`` it is BPE-trained once
over the train split and cached to disk, then reloaded on later runs.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from miditok import REMI, TokenizerConfig
from torch.utils.data import DataLoader, Dataset

from ..data.groove_dataset import (
    ROLAND_TO_GM,
    BeatType,
    GrooveDataPoint,
    GrooveDataset,
)
from ..data.midi import Midi
from ..data.tokenizer import Tokenizer

if TYPE_CHECKING:  # avoid an import cycle at runtime (train imports dataset)
    from .train import TrainConfig

#: Number of distinct genres (primary beat types) used as a conditioning label.
GENRE_VOCAB = len(BeatType)
#: Number of groove types: ``beat`` (0) and ``fill`` (1).
FILL_VOCAB = 2

#: Ordered ``BeatType -> id`` map. Enum iteration order is stable, so the ids are
#: deterministic across runs.
GENRE_TO_ID: dict[BeatType, int] = {beat: i for i, beat in enumerate(BeatType)}
#: ``groove_type -> id`` map (matches ``GrooveType`` literals).
FILL_TO_ID: dict[str, int] = {"beat": 0, "fill": 1}


def build_tokenizer(
    vocab_size: int,
    use_bpe: bool,
    dataset: list[GrooveDataPoint] | None = None,
    cache_path: str | Path | None = None,
) -> REMI:
    """Build (or load) a REMI tokenizer with ``PAD``/``BOS``/``EOS`` specials.

    Without ``use_bpe`` the tokenizer's fixed base vocabulary is returned as-is
    (deterministic, no training) — this is what the smoke test and unit tests use.

    With ``use_bpe`` the tokenizer is BPE-trained to ``vocab_size`` over
    ``dataset``. If ``cache_path`` exists it is loaded from there instead; after a
    fresh train it is saved to ``cache_path`` so subsequent runs reuse it.
    """
    if use_bpe and cache_path is not None and Path(cache_path).is_file():
        return REMI(params=Path(cache_path))

    tokenizer = REMI(TokenizerConfig(special_tokens=["PAD", "BOS", "EOS"]))
    if not use_bpe:
        return tokenizer

    if not dataset:
        raise ValueError("use_bpe=True requires a non-empty dataset to train on")
    Tokenizer(tokenizer).train_tokenizer(dataset, vocab_size=vocab_size)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(Path(cache_path))
    return tokenizer


def _special_id(tokenizer: REMI, token: str) -> int:
    """Vocabulary id of a special token (e.g. ``"BOS"`` -> ``BOS_None`` id)."""
    return tokenizer.vocab[f"{token}_None"]


def encode_datapoint(tokenizer: REMI, dp: GrooveDataPoint) -> list[int]:
    """Tokenize one performance into ``[BOS, ...ids, EOS]`` token ids."""
    seq = tokenizer.encode(dp.midi.score)
    if isinstance(seq, list):  # miditok may return a per-track list
        seq = seq[0]
    return [_special_id(tokenizer, "BOS"), *seq.ids, _special_id(tokenizer, "EOS")]


def tokens_to_midi(tokenizer: REMI, ids: list[int]) -> Midi | None:
    """Decode generated token ids back into a playable drum :class:`Midi`.

    Inverse of :func:`encode_datapoint`: drops ``PAD``/``BOS``, truncates at the
    first ``EOS``, and decodes the rest. The decoded track comes back as a non-drum
    instrument (``is_drum=False``, Roland pitches), so we force every track to a
    drum channel and remap Roland → General MIDI (:data:`ROLAND_TO_GM`) so a GM
    SoundFont triggers the intended kit. Returns ``None`` when nothing decodable
    remains (an empty or all-special sequence).
    """
    pad = _special_id(tokenizer, "PAD")
    bos = _special_id(tokenizer, "BOS")
    eos = _special_id(tokenizer, "EOS")

    clean: list[int] = []
    for token in ids:
        if token == eos:
            break
        if token in (pad, bos):
            continue
        clean.append(token)
    if not clean:
        return None

    score = tokenizer.decode([clean])
    for track in score.tracks:
        track.is_drum = True
    return Midi(score).remap_pitches(ROLAND_TO_GM)


class GrooveTokenDataset(Dataset):
    """Tokenised Groove performances as ``(token_ids, genre_id, fill_id)`` items.

    Clips whose token sequence (including BOS/EOS) exceeds ``max_seq_len`` are
    dropped up-front; the number dropped is warned about once. Each retained item
    is a ``torch.long`` tensor plus its two integer conditioning labels.
    """

    def __init__(
        self,
        tokenizer: REMI,
        datapoints: list[GrooveDataPoint],
        max_seq_len: int,
    ) -> None:
        self.samples: list[tuple[torch.Tensor, int, int]] = []
        dropped = 0
        for dp in datapoints:
            ids = encode_datapoint(tokenizer, dp)
            if len(ids) > max_seq_len:
                dropped += 1
                continue
            self.samples.append(
                (
                    torch.tensor(ids, dtype=torch.long),
                    GENRE_TO_ID[dp.primary_beat_type],
                    FILL_TO_ID[dp.groove_type],
                )
            )
        if dropped:
            warnings.warn(
                f"GrooveTokenDataset: dropped {dropped}/{len(datapoints)} clips "
                f"longer than max_seq_len={max_seq_len}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        return self.samples[index]


def make_collate_fn(pad_token_id: int):
    """Build a collate fn that right-pads token ids with ``pad_token_id``.

    Returns batches as ``{"tokens": [B, T], "genre": [B], "fill": [B]}`` where each
    row of ``tokens`` is padded to the batch's longest sequence.
    """

    def collate_fn(
        batch: list[tuple[torch.Tensor, int, int]],
    ) -> dict[str, torch.Tensor]:
        token_lists, genres, fills = zip(*batch)
        tokens = torch.nn.utils.rnn.pad_sequence(
            list(token_lists), batch_first=True, padding_value=pad_token_id
        )
        return {
            "tokens": tokens,
            "genre": torch.tensor(genres, dtype=torch.long),
            "fill": torch.tensor(fills, dtype=torch.long),
        }

    return collate_fn


def build_dataloaders(
    config: "TrainConfig",
    tokenizer: REMI,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/validation/test dataloaders from the real Groove splits.

    Uses :meth:`GrooveDataset.split` for the official splits and the tokenizer's
    ``PAD`` id for padding. Only the train loader is shuffled.
    """
    dataset = GrooveDataset(config.data_root)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    def loader(split: str, *, shuffle: bool) -> DataLoader:
        points = dataset.split(split)  # type: ignore[arg-type]
        if config.limit_per_split is not None:
            points = points[: config.limit_per_split]
        token_ds = GrooveTokenDataset(tokenizer, points, config.max_seq_len)
        return DataLoader(
            token_ds,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )

    return (
        loader("train", shuffle=True),
        loader("validation", shuffle=False),
        loader("test", shuffle=False),
    )
