import torch
from torch import Tensor

from typing import Optional, Union, Sequence, Literal

from ..utils.python import get_properties

class Metric():
    def __init__(
        self,
        device: str = 'cpu',
        prefix: Optional[str] = None, 
        freq: tuple[str, int] = ('epoch', 1),
    ):
        """
        base metric class, need to be inherited and implement update and reset function, 
        must have properties variable, such as self.properties = ['acc', 'auc']

        Args:
            device: tensor device
            prefix:
                - None: tensorboard log will use var name 'train/auc'
                - str: tensorboard log will use var name f'train/{prefix}_auc'
            freq: frequency about auto log to tensorboard
                - ('epoch', 1): log every epoch
                - ('step', 5): log every 5 steps
        """
        self.device = device
        self.to(device)

        # use for tensorboard logger
        self.prefix = prefix
        self.freq = freq

    def to(self, device: str):
        self.device = device
        for key, val in vars(self).items():
            if isinstance(val, Tensor):
                setattr(self, key, val.to(device))
        return self

    @property
    def res(self) -> dict:
        return {key: getattr(self, key).cpu().item() for key in self.properties}
