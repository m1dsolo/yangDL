from time import sleep

from rich.progress import Progress, BarColumn, MofNCompleteColumn, TaskID

from typing import Sequence

__all__ = [
    'EpochProgress',
]

class EpochProgress(Progress):
    def __init__(self, stage: str, total: int, fold: int, epoch: int):
        super().__init__(transient=True)
        self.columns = (
            '{task.description}: fold: {task.fields[fold]}, epoch: {task.fields[epoch]}',
            BarColumn(),
            MofNCompleteColumn(),
        )
        self.keys = set()

        color = {'train': 'red', 'val': 'blue', 'test': 'green', 'predict': 'pink'}[stage]
        self.task_id = self.add_task(description=f'[{color}]{stage}[/{color}]', total=total, fold=fold, epoch=epoch)

    def update(self, **kwargs):
        for key in kwargs:
            if key not in self.keys:
                self.keys.add(key)
                self.columns = (*self.columns, f'{key}: {{task.fields[{key}]:.3f}}')

        super().update(self.task_id, **kwargs)
        super().advance(self.task_id)
