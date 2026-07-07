"""A small recurrent next-token model over maze token sequences."""

from __future__ import annotations

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
    """Embed token ids, run a (uni-directional) recurrent core, project to logits.

    ``forward`` returns per-timestep logits so the model can be trained as a
    next-token predictor and unrolled autoregressively at generation time. The
    core is never bidirectional: predicting token ``t`` may only depend on tokens
    ``< t``.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        n_layers: int = 2,
        cell_type: str = "rnn",
    ):
        super().__init__()
        if cell_type not in _CELLS:
            raise ValueError(f"cell_type must be one of {sorted(_CELLS)}, got {cell_type!r}")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cell_type = cell_type

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = _CELLS[cell_type](
            embedding_dim, hidden_size, n_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: Tensor, hidden=None) -> tuple[Tensor, object]:
        """Map token ids ``[batch, seq]`` to logits ``[batch, seq, vocab]``.

        ``hidden`` carries the recurrent state between calls (used for
        step-by-step generation); it is returned alongside the logits.
        """
        embedded = self.embedding_layer(x)
        out, hidden = self.rnn(embedded, hidden)
        logits = self.fc(out)
        return logits, hidden
