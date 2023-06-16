import torch
from torch import Tensor

from typing import Optional, Sequence, Literal

from .metric import Metric

class LossMetric(Metric):
    def __init__(
        self,
        prefix: Optional[str] = None,
        freq: tuple[str, int] = ('epoch', 1),
    ):
        """
        loss metric

        Args:
            prefix, freq: see Metric class
        """
        super().__init__(prefix, freq)

        self.add_tensor('_loss', torch.tensor(0, dtype=torch.float))
        self.add_tensor('n', torch.tensor(0, dtype=torch.long))

        self.properties = ['loss']

    @torch.no_grad()
    def update(self, loss: Tensor, n: int | Tensor = 1):
        loss = loss.detach()
        if isinstance(n, int):
            n = torch.tensor(n, device=loss.device)
        elif isinstance(n, Tensor):
            n = n.detach()

        self._loss += loss * n
        self.n += n

    @property
    def loss(self) -> Tensor:
        if self.n == 0:
            return torch.tensor(-1.)
        return self._loss / self.n
