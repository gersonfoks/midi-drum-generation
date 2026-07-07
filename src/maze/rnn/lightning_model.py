"""PyTorch Lightning wrapper: trains :class:`RNNModel` as a next-token predictor.

Training uses teacher forcing — the model sees the ground-truth prefix and
predicts the next token at every position. The recurrent core must stay
uni-directional so position ``t`` only ever conditions on positions ``< t``.
"""

from __future__ import annotations

import lightning as pl
import torch
from torch import Tensor, nn

from ..dataset import BOS_TOKEN, VOCAB_SIZE


class LightningRnnModel(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: Tensor, hidden=None) -> tuple[Tensor, object]:
        return self.model(x, hidden)

    def shared_step(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Teacher-forced next-token loss and accuracy over the whole sequence."""
        x, y = batch[:, :-1], batch[:, 1:]
        logits, _ = self(x)
        loss = self.loss_fn(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        accuracy = (logits.argmax(dim=-1) == y).float().mean()
        return loss, accuracy

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss, accuracy = self.shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", accuracy, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss, accuracy = self.shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", accuracy, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def generate_batch(self, length: int, count: int, temperature: float = 1.0) -> Tensor:
        """Sample ``count`` sequences of ``length`` cell tokens in one batched pass.

        All ``count`` sequences are decoded together: each of the ``length`` steps
        runs a single forward over the whole batch (``length`` forward passes total,
        not ``count * length``). Tokens are sampled from the model's distribution —
        ``temperature`` scales the logits, so the neutral ``1.0`` samples from the
        real learned distribution. Returns a ``[count, length]`` long tensor of
        generated cell tokens (the seeding BOS is not included), ready for
        :func:`~src.maze.dataset.tokens_to_maze`.
        """
        was_training = self.training
        self.eval()
        input_token = torch.full((count, 1), BOS_TOKEN, dtype=torch.long, device=self.device)
        hidden = None
        tokens: list[Tensor] = []

        for _ in range(length):
            logits, hidden = self(input_token, hidden)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            input_token = torch.multinomial(probs, num_samples=1)  # [count, 1]
            tokens.append(input_token)

        if was_training:
            self.train()
        return torch.cat(tokens, dim=1)

    def generate(self, length: int, temperature: float = 1.0) -> list[int]:
        """Sample a single sequence of ``length`` cell tokens (see
        :meth:`generate_batch`)."""
        return self.generate_batch(length, count=1, temperature=temperature)[0].tolist()
