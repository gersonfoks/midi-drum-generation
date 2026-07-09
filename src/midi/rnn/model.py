"""A small recurrent next-token model over MIDI token sequences.

The model is conditioned on two pieces of track info — the *genre* and whether the
clip is a *fill*. Each is embedded and concatenated onto every token embedding
before the recurrent core, so the condition is visible at every timestep and the
core stays cell-agnostic (the same code path works for RNN/LSTM/GRU).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

#: Supported recurrent cell types, mapping a name to its ``nn`` module. Typed as
#: plain modules because the concrete cells share the ``(input_size, hidden_size,
#: num_layers, ...)`` signature rather than ``nn.RNNBase``'s ``mode``-first one.
_CELLS: dict[str, type[nn.Module]] = {
    "rnn": nn.RNN,
    "lstm": nn.LSTM,
    "gru": nn.GRU,
}


class RNNModel(nn.Module):
    """Embed token ids + conditioning, run a recurrent core, project to logits.

    ``forward`` returns per-timestep logits so the model can be trained as a
    next-token predictor and unrolled autoregressively at generation time. The
    core is never bidirectional: predicting token ``t`` may only depend on tokens
    ``< t`` (and the conditioning, which is constant over the sequence).
    """

    def __init__(
        self,
        vocab_size: int,
        genre_vocab: int,
        fill_vocab: int,
        embedding_dim: int = 64,
        genre_embedding_dim: int = 16,
        fill_embedding_dim: int = 8,
        hidden_size: int = 128,
        n_layers: int = 2,
        cell_type: str = "lstm",
        pad_token_id: int = 0,
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        if cell_type not in _CELLS:
            raise ValueError(f"cell_type must be one of {sorted(_CELLS)}, got {cell_type!r}")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cell_type = cell_type
        self.pad_token_id = pad_token_id
        self.residual = residual

        self.embedding_layer = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.genre_embedding = nn.Embedding(genre_vocab, genre_embedding_dim)
        self.fill_embedding = nn.Embedding(fill_vocab, fill_embedding_dim)

        rnn_input_size = embedding_dim + genre_embedding_dim + fill_embedding_dim
        cell = _CELLS[cell_type]
        if residual:
            # Build the stack by hand as single-layer cells so we can add a skip
            # connection around each layer. Residuals ease gradient flow through a
            # deep stack, which helps when training on longer sequences. The first
            # layer's input width (``rnn_input_size``) differs from ``hidden_size``,
            # so its skip is dropped in ``forward`` (see the shape guard there).
            self.rnn_layers = nn.ModuleList(
                cell(
                    rnn_input_size if i == 0 else hidden_size,
                    hidden_size,
                    1,
                    batch_first=True,
                )
                for i in range(n_layers)
            )
            self.inter_dropout = nn.Dropout(dropout)
            self.rnn = None
        else:
            # ``dropout`` only applies between stacked layers, so it is a no-op (and
            # warns) for a single layer; guard against that.
            self.rnn_layers = None
            self.rnn = cell(
                rnn_input_size,
                hidden_size,
                n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, x: Tensor, genre: Tensor, fill: Tensor, hidden=None
    ) -> tuple[Tensor, object]:
        """Map token ids ``[batch, seq]`` (+ conditioning) to logits.

        ``genre`` and ``fill`` are ``[batch]`` id tensors; their embeddings are
        broadcast across the sequence and concatenated onto each token embedding.
        ``hidden`` carries the recurrent state between calls (used for
        step-by-step generation) and is returned alongside the logits. Its concrete
        type is opaque and mode-dependent (a single cell's state without residuals,
        a per-layer list with them), so callers must only pass it back as-is.
        """
        embedded = self.embedding_layer(x)  # [B, T, E]
        cond = self._condition(genre, fill, seq_len=embedded.size(1))  # [B, T, C]
        rnn_in = torch.cat([embedded, cond], dim=-1)
        if self.residual:
            out, hidden = self._forward_residual(rnn_in, hidden)
        else:
            assert self.rnn is not None  # set whenever residual is False
            out, hidden = self.rnn(rnn_in, hidden)
        logits = self.fc(out)
        return logits, hidden

    def _forward_residual(self, rnn_in: Tensor, hidden) -> tuple[Tensor, list]:
        """Run the per-layer stack, adding a skip connection around each layer.

        The skip is applied only where a layer's input and output widths match
        (every layer after the first, and the first too if the conditioned input
        already has width ``hidden_size``). Dropout is applied between layers, never
        after the last. ``hidden`` is a per-layer list of states (or ``None``).
        """
        assert self.rnn_layers is not None  # set whenever residual is True
        states = hidden if hidden is not None else [None] * len(self.rnn_layers)
        new_states: list = []
        layer_input = rnn_in
        last = len(self.rnn_layers) - 1
        for i, layer in enumerate(self.rnn_layers):
            out, state = layer(layer_input, states[i])
            if out.shape[-1] == layer_input.shape[-1]:
                out = out + layer_input  # residual skip
            if i != last:
                out = self.inter_dropout(out)
            layer_input = out
            new_states.append(state)
        return layer_input, new_states

    def _condition(self, genre: Tensor, fill: Tensor, seq_len: int) -> Tensor:
        """Broadcast the genre + fill embeddings to ``[batch, seq_len, C]``."""
        cond = torch.cat([self.genre_embedding(genre), self.fill_embedding(fill)], dim=-1)
        return cond.unsqueeze(1).expand(-1, seq_len, -1)
