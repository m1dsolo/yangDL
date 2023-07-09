import torch
from torch import Tensor
from copy import deepcopy

from typing import Optional, Union, Sequence, Literal

from ..utils.python import get_properties

class Metric():
    def __init__(
        self,
        prefix: Optional[str] = None, 
        freq: tuple[str, int] = ('epoch', 1),
    ):
        """
        base metric class, need to be inherited and implement update function, 
        must have properties variable, such as self.properties = ['acc', 'auc']

        Args:
            prefix:
                - None: tensorboard log will use var name 'train/auc'
                - str: tensorboard log will use var name f'train/{prefix}_auc'
            freq: frequency about auto log to tensorboard
                - ('epoch', 1): log every epoch
                - ('step', 5): log every 5 steps
        """
        # use for tensorboard logger
        self.prefix = prefix
        self.freq = freq

        self.named_tensors = {}

    def add_tensor(
        self, 
        name: str, 
        tensor: Tensor, 
    ) -> None:
        """
        add tensor variable, can auto handle default value and reset, used in reset() function.

        Args:
            name: variable name
            tensor: tensor default value
        """
        self.named_tensors[name] = tensor
        setattr(self, name, deepcopy(tensor))

    def to(self, device: str):
        """
        move all tensor and Metric's tensor to device
        """
        self.named_tensors = {name: tensor.to(device) for name, tensor in self.named_tensors.items()}

        for name, val in vars(self).items():
            if isinstance(val, Tensor):
                setattr(self, name, val.to(device))
            if isinstance(val, Metric):
                getattr(self, name).to(device)

    def reset(self) -> None:
        """
        reset all tensor and Metric when epoch loop finished.
        """
        for name, tensor in self.named_tensors.items():
            setattr(self, name, deepcopy(tensor))

        for name, val in vars(self).items():
            if isinstance(val, Metric):
                getattr(self, name).reset()

    @property
    def res(self) -> dict:
        """
        will return all {property: value} of the Metric
        """
        return {key: getattr(self, key).cpu().item() for key in self.properties}

