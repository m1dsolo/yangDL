import torch, os, copy, datetime
import numpy as np
from collections import defaultdict

from torch import Tensor
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from typing import Optional

from rich import print

from . import env
from .model_module import ModelModule
from .loader_module import LoaderModule
from .logger import Logger
from .early_stop import EarlyStop
from ..metric import Metric
from .ui import EpochProgress

from ..utils.os import mkdir
from ..utils.io import write_dict
from ..utils.dl import get_model_devices
from ..utils.helper import WithNone
from ..utils.python import method_is_overrided_in_subclass
from ..utils.func import merge_dict
from ..utils.device import get_max_gpu

__all__ = [
    'Trainer',
]

class Trainer():
    def __init__(self,
                 model_module: ModelModule,
                 loader_module: LoaderModule,
                 device: Optional[str] = None,
                 seed: Optional[int] = None,
                 early_stop_params: Optional[dict] = None,
                 save_all_ckpt: bool = False,
                 benchmark: bool = False,
                 deterministic: bool = True,
                 val_first: bool = False,
                 ):
        """
        function:
        1. handle fold, epoch loop
        2. mixed precision
        3. save_ckpt, load_ckpt

        Args:
            model_module: ModelModule
            loader_module: LoaderModule
            device: pytorch device
            seed: for reproduction
            early_stop_params: is a params dict to initialize EarlyStop, 
                example:
                {
                    'patience': 25, # if loss not decrease in 25 epochs, will early stop.
                    'min_stop_epoch': 25, # min epoch to early stop
                    'max_stop_epoch': 100, # max epoch to early stop
                }
            save_all_ckpt:
                True: save all ckpt
                False: only save best ckpt
            benchmark: torch.backends.cudnn.benchmark
            deterministic: torch.backends.cudnn.deterministic
            val_first:
                True: begin with epoch 0 to val
                False: begin with epoch 1 to train
        """

        self.model_module = model_module
        self.loader_module = loader_module
        self.device = device or f'cuda:{get_max_gpu()[0]}'
        self.seed = seed
        self.early_stop = EarlyStop(**early_stop_params) if early_stop_params is not None else None
        self.logger = None
        self.save_all_ckpt = save_all_ckpt
        self.val_first = val_first

        self.train = method_is_overrided_in_subclass(loader_module.train_loader)
        self.val = method_is_overrided_in_subclass(loader_module.val_loader)
        self.test = method_is_overrided_in_subclass(loader_module.test_loader)

        # f'{stage}_res': {'loss': {'loss': [0.2, 0.4, 0.3]}, 'metric': {'acc': [0.7, 0.5, 0.6], 'auc': [0.6, 0.5, 0.6]}}
        if self.train:
            self.train_res: defaultdict[str, dict] = defaultdict(dict)
            self.best_train_named_metrics: dict[str, Metric] = {}
        if self.val:
            self.val_res: defaultdict[str, dict] = defaultdict(dict)
            self.best_val_named_metrics: dict[str, Metric] = {}
        if self.test:
            self.test_res = defaultdict(dict)

        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic

    @staticmethod
    def _wrapper(func):
        def wrapper(*args, **kwargs):
            start = datetime.datetime.now()
            res = func(*args, **kwargs)
            print(f'total time: {datetime.datetime.now() - start}')
            if EXP_NAME := env.get("EXP_NAME") is not None:
                print(f'EXP_NAME: {env.get("EXP_NAME")}')
            return res

        return wrapper

    @_wrapper
    def do(
        self, 
    ) -> None:
        """
        1. train
        2. train, val
        3. train, val, test
        4. train, val, predict(kaggle)
        5. train, val, test, predict
        """

        for fold, (_, train_loader, val_loader, test_loader, predict_loader) in enumerate(zip(self.model_module, self.loader_module.train_loader(), self.loader_module.val_loader(), self.loader_module.test_loader(), self.loader_module.predict_loader()), 1):
            for name, model in self.model_module.named_models.items():
                setattr(self.model_module, name, model.to(self.device))
            for name, metric in self.model_module.named_metrics.items():
                metric.to(self.device)

            self.logger = Logger(fold)
            if self.early_stop is not None:
                self.early_stop.init()

            self._do_fold(fold, train_loader, val_loader, test_loader, predict_loader)
        res = self._save_res()
        print(res)

    def _do_fold(
        self, 
        fold: Optional[int], 
        train_loader: Optional[DataLoader] = None, 
        val_loader: Optional[DataLoader] = None, 
        test_loader: Optional[DataLoader] = None,
        predict_loader: Optional[DataLoader] = None,
    ) -> None:

        if train_loader is None and val_loader is None and test_loader is None:
            self._do_epoch('predict', predict_loader, fold, -1)
            return

        if train_loader:
            for epoch in range(0 if self.val_first else 1, self.early_stop.max_stop_epoch + 1):
                if self.save_all_ckpt:
                    self._save_ckpt(fold, epoch)

                if epoch:
                    train_named_metrics = self._do_epoch('train', train_loader, fold, epoch)

                if val_loader:
                    val_named_metrics = self._do_epoch('val', val_loader, fold, epoch)

                    self.early_stop(val_named_metrics[env.get('LOSS_VAR_NAME')].loss, epoch)
                    if not self.save_all_ckpt and self.early_stop.best_epoch == epoch:
                        self._save_ckpt(fold, None)
                        if epoch:
                            self.best_train_named_metrics = train_named_metrics

                        self.best_val_named_metrics = val_named_metrics
                    if self.early_stop.early_stop:
                        break

        if train_loader:
            self._update_res(self.train_res, self.best_train_named_metrics)
        if val_loader:
            self._update_res(self.val_res, self.best_val_named_metrics)
        if test_loader:
            self._load_ckpt(fold, self.early_stop.best_epoch if self.save_all_ckpt else None)
        if test_loader or predict_loader:
            named_metrics = self._do_epoch('test', test_loader, fold, -1)
            self._update_res(self.test_res, named_metrics)
        if predict_loader:
            self._do_epoch('predict', predict_loader, fold, -1)

    def _do_epoch(
        self,
        stage: str,
        loader: DataLoader,
        fold: int,
        epoch: int
    ) -> dict[str, Metric]:

        env.set(STAGE=stage, FOLD=fold, EPOCH=epoch)
        with WithNone() if stage == 'train' else torch.no_grad(), EpochProgress(stage, len(loader), fold, epoch) as progress:
            for model in self.model_module.named_models.values():
                model.train() if stage == 'train' else model.eval()

            for step, batch in enumerate(loader, 1):
                step_res = self._do_step(stage, step, batch) or {}
                progress.update(**step_res)

                self._log(stage, self.model_module.named_metrics, step=epoch * len(loader) + step)
            self._log(stage, self.model_module.named_metrics, epoch=epoch)

            getattr(self.model_module, f'{stage}_epoch_end')()

            named_metrics = copy.deepcopy(self.model_module.named_metrics)
            for metric in self.model_module.named_metrics.values():
                metric.reset()
            return named_metrics

    def _do_step(
        self, 
        stage: str, 
        step: int, 
        batch: list
    ) -> None:

        with autocast():
            batch = self._batch_to_device(batch)
            return getattr(self.model_module, f'{stage}_step')(batch)

    def _save_ckpt(
        self, 
        fold: Optional[int] = None, 
        epoch: Optional[int] = None
    ) -> None:

        ckpt_path = env.get('CKPT_PATH')
        if fold is not None:
            ckpt_path = os.path.join(ckpt_path, str(fold))
        mkdir(ckpt_path)
        ckpt_name = os.path.join(ckpt_path, 'ckpt.pth' if epoch is None else f'ckpt_{epoch}.pth')

        self.model_module.save_ckpt(ckpt_name)

    def _load_ckpt(
        self, 
        fold: Optional[int] = None, 
        epoch: Optional[int] = None
    ) -> None:

        ckpt_path = env.get('CKPT_PATH')
        if fold is not None:
            ckpt_path = os.path.join(ckpt_path, str(fold))
        ckpt_name = os.path.join(ckpt_path, 'ckpt.pth' if epoch is None else f'ckpt_{epoch}.pth')

        self.model_module.load_ckpt(ckpt_name)

    def _save_res(
        self
    ) -> dict:

        metric_path = env.get('METRIC_PATH')
        write_dict(os.path.join(metric_path, f'early_stop.json'), self.early_stop.to_dict())

        res = {}
        for stage in ('train', 'val', 'test'):
            if hasattr(self, f'{stage}_res'):
                stage_res = getattr(self, f'{stage}_res')
                res[stage] = {}
                for metric_name, d in stage_res.items():
                    res[stage][metric_name] = {}
                    for property_name, arr in d.items():
                        stage_res[metric_name][property_name] = {}
                        stage_res[metric_name][property_name]['data'] = arr
                        stage_res[metric_name][property_name]['mean'] = sum(arr) / len(arr)
                        stage_res[metric_name][property_name]['max'] = max(arr)
                        stage_res[metric_name][property_name]['min'] = min(arr)
                        stage_res[metric_name][property_name]['std'] = np.array(arr).std()

                        res[stage][metric_name][property_name] = stage_res[metric_name][property_name]['mean']

                write_dict(os.path.join(metric_path, f'{stage}.json'), stage_res)

        return res

    def _check_train(self):
        assert self.save_all_ckpt is True

    def _update_res(self, res: dict[str, dict[str, list]], named_metrics: dict[str, Metric]):
        for name, metric in named_metrics.items():
            res[name] = merge_dict(res[name], metric.res)

    def _log(self, stage: str, named_metrics: dict[str, Metric], epoch: Optional[int] = None, step: Optional[int] = None):
        if epoch is not None and epoch < 0 or step is not None and step < 0:
            if stage != 'test':
                return
            epoch = 0
        for name, metric in named_metrics.items():
            if epoch is not None and metric.freq[0] == 'epoch' and epoch % metric.freq[1] == 0:
                self.logger.log(metric, stage, epoch)
            if step is not None and metric.freq[0] == 'step' and step % metric.freq[1] == 0:
                self.logger.log(metric, stage, step)
    
    def _batch_to_device(self, batch):
        """
        incomplete consideration
        """
        if isinstance(batch, Tensor):
            batch = batch.to(self.device)
        if isinstance(batch, dict):
            batch = {key: (val.to(self.device) if isinstance(val, torch.Tensor) else val)for key, val in batch.items()}
        else:
            batch = [(item.to(self.device) if isinstance(item, torch.Tensor) else item) for item in batch]
        return batch
