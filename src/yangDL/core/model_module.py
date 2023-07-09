import torch, os

from torch import nn
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from rich import print

from . import env
from ..metric import Metric

from ..utils.python import clear_rich

__all__ = [
    'ModelModule',
]

class ModelModule():
    def __init__(self):
        pass

    def __iter__(self):
        yield

    def train_step(self, batch) -> None:
        pass

    def val_step(self, batch) -> None:
        pass

    def test_step(self, batch) -> None:
        pass

    def predict_step(self, batch) -> dict:
        pass

    def train_epoch_end(self) -> None:
        pass

    def val_epoch_end(self) -> None:
        pass

    def test_epoch_end(self) -> None:
        pass

    def predict_epoch_end(self) -> None:
        pass

    @property
    def named_models(self) -> dict[str, nn.Module]:
        return dict(filter(lambda kv: Module in kv[1].__class__.__mro__, self.__dict__.items()))

    @property
    def named_metrics(self) -> dict[str, Metric]:
        return dict(filter(lambda kv: Metric in kv[1].__class__.__mro__, self.__dict__.items()))

    @property
    def named_optimizers(self) -> dict[str, Optimizer]:
        return dict(filter(lambda kv: Optimizer in kv[1].__class__.__mro__, self.__dict__.items()))

    @property
    def named_schedulers(self) -> dict[str, _LRScheduler]:
        return dict(filter(lambda kv: _LRScheduler in kv[1].__class__.__mro__, self.__dict__.items()))

    def to(self, device: str):
        """
        move all models and metrics in ModelModule to device
        """

        for model in self.named_models.values():
            model = model.to(device)
        for metric in self.named_metrics.values():
            metric = metric.to(device)

        return self

    @property
    def device(self) -> torch.device:
        """
        only support one gpu now
        """

        for model in self.named_models.values():
            for param in model.parameters():
                return param.device

    def print(self, **kwargs) -> None:
        """
        use to print res to terminal and log_file

        Examples:
        >>> self.print(loss=self.loss.loss, auc=self.metric.auc)
        {stage}: fold: {fold}, epoch: {epoch}: loss: 0.3, auc: 0.7
        """

        stage, fold, epoch = env.get('STAGE', 'FOLD', 'EPOCH')
        color = {'train': 'red', 'val': 'blue', 'test': 'green', 'predict': 'pink'}[stage]

        s = f'[{color}]{stage}[/{color}]: fold: {fold}, epoch: {epoch}: '
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                val = val.cpu().item()
            s += f'{key}: [{color}]{val: .3f}[/{color}], '
        s = s[:-2]

        print(s)
        if log_file_name:= env.get('LOG_FILE_NAME'):
            log_path = env.get('LOG_PATH')
            with open(os.path.join(log_path, str(fold), log_file_name), 'a') as f:
                f.write(f'{clear_rich(s)}\n')

    def save_ckpt(self, ckpt_name: str):
        """
        save all model in ModelModule to ckpt_name, can be overridden to customize saves

        TODO: save optimizers, lr_schedulers
        """

        ckpt = {}
        for name, model in self.named_models.items():
            ckpt[name] = model.state_dict()

        torch.save(ckpt, ckpt_name)

    def load_ckpt(self, ckpt_name: str):
        """
        can be overridden to customize loads
        """
        
        ckpt = torch.load(ckpt_name)
        for name, state_dict in ckpt.items():
            getattr(self, name).load_state_dict(state_dict)
