from torch.utils.data import DataLoader

from typing import Generator

__all__ = [
    'LoaderModule',
]

class LoaderModule():
    def __init__(self):
        pass

    def train_loader(self) -> Generator[DataLoader, None, None]:
        while True:
            yield None

    def val_loader(self) -> Generator[DataLoader, None, None]:
        while True:
            yield None

    def test_loader(self) -> Generator[DataLoader, None, None]:
        while True:
            yield None

    def predict_loader(self) -> DataLoader:
        while True:
            yield None
