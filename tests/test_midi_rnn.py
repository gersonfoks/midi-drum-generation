"""Tests for the conditional MIDI-groove RNN pipeline.

The unit tests (model / collate / shared_step / generation) are data-free so they
run in CI without the Groove dataset. The end-to-end ``smoke_test`` needs the raw
MIDI, so it is skipped when the dataset is absent (mirroring
``tests/test_groove_dataset.py``).
"""

from __future__ import annotations

import pytest
import torch

from src.midi.data.groove_dataset import DEFAULT_ROOT
from src.midi.rnn.app import _smoke_test
from src.midi.rnn.dataset import (
    FILL_VOCAB,
    GENRE_VOCAB,
    build_tokenizer,
    encode_datapoint,
    make_collate_fn,
    tokens_to_midi,
)
from src.midi.rnn.lightning_model import LightningRnnModel
from src.midi.rnn.model import RNNModel

VOCAB_SIZE = 40
PAD_ID = 0
BOS_ID = 1


def _build_model(
    cell_type: str = "gru", *, n_layers: int = 1, residual: bool = False
) -> LightningRnnModel:
    model = RNNModel(
        vocab_size=VOCAB_SIZE,
        genre_vocab=GENRE_VOCAB,
        fill_vocab=FILL_VOCAB,
        embedding_dim=8,
        genre_embedding_dim=4,
        fill_embedding_dim=4,
        hidden_size=16,
        n_layers=n_layers,
        cell_type=cell_type,
        pad_token_id=PAD_ID,
        residual=residual,
    )
    return LightningRnnModel(model, vocab_size=VOCAB_SIZE, pad_token_id=PAD_ID)


# --- model ----------------------------------------------------------------


@pytest.mark.parametrize("cell_type", ["rnn", "lstm", "gru"])
def test_forward_returns_per_timestep_logits(cell_type: str):
    model = _build_model(cell_type).model
    x = torch.zeros(3, 10, dtype=torch.long)
    genre = torch.zeros(3, dtype=torch.long)
    fill = torch.ones(3, dtype=torch.long)
    logits, _ = model(x, genre, fill)
    assert logits.shape == (3, 10, VOCAB_SIZE)


def test_model_rejects_unknown_cell_type():
    with pytest.raises(ValueError):
        RNNModel(VOCAB_SIZE, GENRE_VOCAB, FILL_VOCAB, cell_type="transformer")


@pytest.mark.parametrize("cell_type", ["rnn", "lstm", "gru"])
def test_residual_forward_returns_per_timestep_logits(cell_type: str):
    # A multi-layer residual stack must still map to [B, T, vocab] for every cell.
    model = _build_model(cell_type, n_layers=3, residual=True).model
    x = torch.zeros(3, 10, dtype=torch.long)
    genre = torch.zeros(3, dtype=torch.long)
    fill = torch.ones(3, dtype=torch.long)
    logits, hidden = model(x, genre, fill)
    assert logits.shape == (3, 10, VOCAB_SIZE)
    assert isinstance(hidden, list) and len(hidden) == 3  # per-layer state


def test_residual_generation_threads_hidden_state():
    # generate_batch feeds the returned (per-layer) hidden back in each step.
    model = _build_model("lstm", n_layers=2, residual=True)
    batch = model.generate_batch(
        length=8,
        genre_ids=torch.tensor([0, 3]),
        fill_ids=torch.tensor([1, 0]),
        bos_token_id=BOS_ID,
    )
    assert batch.shape == (2, 8)


# --- collate --------------------------------------------------------------


def test_collate_right_pads_to_batch_max():
    collate = make_collate_fn(pad_token_id=PAD_ID)
    batch = [
        (torch.tensor([1, 5, 6, 2]), 3, 1),
        (torch.tensor([1, 7, 2]), 4, 0),
    ]
    out = collate(batch)
    assert out["tokens"].shape == (2, 4)  # padded to the longest (4)
    assert out["tokens"][1].tolist() == [1, 7, 2, PAD_ID]  # shorter row right-padded
    assert out["genre"].tolist() == [3, 4]
    assert out["fill"].tolist() == [1, 0]


# --- shared_step ----------------------------------------------------------


def test_shared_step_gives_finite_loss_and_ignores_pad():
    model = _build_model()
    tokens = torch.randint(1, VOCAB_SIZE, (2, 12))
    tokens[:, -3:] = PAD_ID  # trailing padding on both rows
    batch = {
        "tokens": tokens,
        "genre": torch.randint(0, GENRE_VOCAB, (2,)),
        "fill": torch.randint(0, FILL_VOCAB, (2,)),
    }
    loss, accuracy = model.shared_step(batch)
    assert torch.isfinite(loss)
    assert 0.0 <= float(accuracy) <= 1.0


def test_shared_step_all_pad_targets_gives_zero_accuracy():
    # If every target is padding, accuracy is defined (no div-by-zero) and 0.
    model = _build_model()
    tokens = torch.full((2, 6), PAD_ID, dtype=torch.long)
    tokens[:, 0] = BOS_ID
    batch = {
        "tokens": tokens,
        "genre": torch.zeros(2, dtype=torch.long),
        "fill": torch.zeros(2, dtype=torch.long),
    }
    _, accuracy = model.shared_step(batch)
    assert float(accuracy) == 0.0


# --- generation -----------------------------------------------------------


def test_generate_batch_shape_and_conditioning():
    model = _build_model()
    genre_ids = torch.tensor([0, 5, 17])
    fill_ids = torch.tensor([1, 0, 1])
    batch = model.generate_batch(length=9, genre_ids=genre_ids, fill_ids=fill_ids, bos_token_id=BOS_ID)
    assert batch.shape == (3, 9)
    assert batch.dtype == torch.long
    assert batch.min() >= 0 and batch.max() < VOCAB_SIZE


def test_generate_single_returns_length():
    model = _build_model()
    tokens = model.generate(length=7, genre_id=2, fill_id=1, bos_token_id=BOS_ID)
    assert len(tokens) == 7


# --- decode to audio-ready MIDI -------------------------------------------


def test_tokens_to_midi_empty_returns_none():
    tokenizer = build_tokenizer(vocab_size=1000, use_bpe=False)
    specials = [
        tokenizer.vocab["PAD_None"],
        tokenizer.vocab["BOS_None"],
        tokenizer.vocab["EOS_None"],
    ]
    assert tokens_to_midi(tokenizer, specials) is None


def test_tokens_to_midi_truncates_at_eos():
    # Everything after the first EOS is ignored: with only specials before it, the
    # cleaned sequence is empty even though real tokens follow -> None.
    tokenizer = build_tokenizer(vocab_size=1000, use_bpe=False)
    bos, eos = tokenizer.vocab["BOS_None"], tokenizer.vocab["EOS_None"]
    ids = [bos, eos, 3, 4, 5, 6]  # tokens after EOS must be dropped
    assert tokens_to_midi(tokenizer, ids) is None


@pytest.mark.skipif(
    not (DEFAULT_ROOT / "info.csv").is_file(),
    reason="Groove dataset not present under data/raw/groove",
)
def test_tokens_to_midi_yields_drum_track():
    from src.midi.data.groove_dataset import GrooveDataset

    tokenizer = build_tokenizer(vocab_size=1000, use_bpe=False)
    dp = GrooveDataset(DEFAULT_ROOT)[0]
    midi = tokens_to_midi(tokenizer, encode_datapoint(tokenizer, dp))
    assert midi is not None
    assert midi.score.tracks, "decoded MIDI has no tracks"
    # Every track must be a drum track so a GM SoundFont plays the kit, not a piano.
    assert all(track.is_drum for track in midi.score.tracks)


# --- end to end -----------------------------------------------------------


@pytest.mark.skipif(
    not (DEFAULT_ROOT / "info.csv").is_file(),
    reason="Groove dataset not present under data/raw/groove",
)
def test_smoke_test_runs():
    _smoke_test()
