"""PyTorch Lightning wrapper: trains :class:`RNNModel` as a next-token predictor.

Training uses teacher forcing — the model sees the ground-truth prefix and
predicts the next token at every position. Sequences are padded per batch, so the
loss and accuracy ignore padded target positions (``ignore_index``). The
recurrent core stays uni-directional so position ``t`` only conditions on
positions ``< t`` (plus the constant genre/fill conditioning).
"""

from __future__ import annotations

import lightning as pl
import torch
from torch import Tensor, nn


class LightningRnnModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        pad_token_id: int,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.save_hyperparameters(ignore=["model"])

    def forward(
        self, x: Tensor, genre: Tensor, fill: Tensor, hidden=None
    ) -> tuple[Tensor, object]:
        return self.model(x, genre, fill, hidden)

    def shared_step(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Teacher-forced next-token loss and accuracy, ignoring padded targets."""
        tokens, genre, fill = batch["tokens"], batch["genre"], batch["fill"]
        x, y = tokens[:, :-1], tokens[:, 1:]
        logits, _ = self(x, genre, fill)
        loss = self.loss_fn(logits.reshape(-1, self.vocab_size), y.reshape(-1))

        mask = y != self.pad_token_id
        correct = (logits.argmax(dim=-1) == y) & mask
        accuracy = correct.sum().float() / mask.sum().clamp(min=1)
        return loss, accuracy

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, accuracy = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", accuracy, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, accuracy = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", accuracy, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def generate_batch(
        self,
        length: int,
        genre_ids: Tensor,
        fill_ids: Tensor,
        bos_token_id: int,
        temperature: float = 1.0,
    ) -> Tensor:
        """Sample sequences conditioned on ``genre_ids`` / ``fill_ids``.

        One sequence is generated per element of ``genre_ids`` (which must match
        ``fill_ids`` in length). Each of the ``length`` steps runs a single forward
        over the whole batch, carrying the recurrent ``hidden`` state, so this is
        ``length`` forward passes total. Tokens are sampled from the temperature-
        scaled distribution (``1.0`` samples the learned distribution). The seeding
        BOS is not included in the returned ``[count, length]`` long tensor.
        """
        was_training = self.training
        self.eval()
        genre_ids = genre_ids.to(self.device)
        fill_ids = fill_ids.to(self.device)
        count = genre_ids.size(0)

        input_token = torch.full(
            (count, 1), bos_token_id, dtype=torch.long, device=self.device
        )
        hidden = None
        tokens: list[Tensor] = []
        for _ in range(length):
            logits, hidden = self(input_token, genre_ids, fill_ids, hidden)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            input_token = torch.multinomial(probs, num_samples=1)  # [count, 1]
            tokens.append(input_token)

        if was_training:
            self.train()
        return torch.cat(tokens, dim=1)

    def generate(
        self,
        length: int,
        genre_id: int,
        fill_id: int,
        bos_token_id: int,
        temperature: float = 1.0,
    ) -> list[int]:
        """Sample a single sequence for one ``(genre, fill)`` (see
        :meth:`generate_batch`)."""
        genre_ids = torch.tensor([genre_id], dtype=torch.long)
        fill_ids = torch.tensor([fill_id], dtype=torch.long)
        return self.generate_batch(
            length, genre_ids, fill_ids, bos_token_id, temperature=temperature
        )[0].tolist()
