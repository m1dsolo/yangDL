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
        self.writer = SummaryWriter(os.path.join(env.get('LOG_PATH'), str(fold)))

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

        for name, val in metric.res.items():
            if metric.prefix is not None:
                name = f'{metric.prefix}_{name}'
            self.writer.add_scalar(f'{name}/{stage}', val, step)

    def __del__(self):
        self.writer.close()
