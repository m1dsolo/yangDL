import os

from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
from torch import Tensor

from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn

from typing import Optional, Sequence

from ..metric import Metric

from . import env
from ..utils.os import mkdir
from ..utils.python import get_properties

__all__ = [
    'Logger',
]

class Logger():
    def __init__(self,
                 fold: int,
                 ):
        log_path = env.get('LOG_PATH')
        if log_path:
            self.writer = SummaryWriter(os.path.join(log_path, str(fold)))
        else:
            self.writer = None

    def log(self, 
            metric: Metric,
            stage: str,
            step: int,
            ):
        """
        log metric to tensorboard

        Args:
            metric: Metric
            stage: in ('train', 'val', 'test')
            step: tensorboard scalar step
        """
        assert self.writer is not None

        for name, val in metric.res.items():
            if metric.prefix is not None:
                name = f'{metric.prefix}_{name}'
            self.writer.add_scalar(f'{name}/{stage}', val, step)

    def __del__(self):
        if self.writer is not None:
            self.writer.close()
