import os, torch, timm, cv2
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import functional as F
from rich import print

from sklearn.model_selection import StratifiedKFold

import albumentations as A
from albumentations.pytorch import ToTensorV2

from yangDL import Trainer, ModelModule, LoaderModule, env
from yangDL.metric import LossMetric, ClsMetric
from yangDL.utils.io import write_dict
from yangDL.utils.os import mkdir

SEED = 1

DATA_PATH = '.'
EXP_PATH = './res/tmp'
env.set(EXP_PATH=EXP_PATH)

PRED_PATH = './pred/tmp'
mkdir(PRED_PATH)

model_hparams = {
    'model': {
        'model_name': 'convnext_small',
        'pretrained': True,
        'in_chans': 1,
        'num_classes': 10,
    },
    'optimizer': {
        'type': 'Adam',
        'params': {
            'lr': 1e-4,
            'weight_decay': 0,
            'betas': (0.9, 0.999),
        },
    },
}

loader_hparams = {
    'seed': SEED,
    'batch_size': 256,
    'num_workers': 4,
}

trainer_hparams = {
    'seed': SEED,
    'early_stop_params': {
        'patience': 10,
        'min_stop_epoch': 10,
        'max_stop_epoch': 100,
    },
    'benchmark': True,
    'deterministic': True,
}

class MyModelModule(ModelModule):
    def __init__(self,
                 hparams: dict,
                 ):

        super().__init__()

        self.hparams = hparams

        self.criterion = nn.CrossEntropyLoss()

        self.loss = LossMetric()
        self.metric = ClsMetric(num_classes=10)

    def __iter__(self):
        while True:
            self.model = timm.create_model(**self.hparams['model'])
            self.preds = []

            params = filter(lambda p: p.requires_grad, self.model.parameters())
            if self.hparams['optimizer']['type'] == 'Adam':
                self.optimizer = Adam(params, **self.hparams['optimizer']['params'])
            yield

    def train_step(self, batch):
        loss = self._step(batch)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {'loss': loss, 'acc': self.metric.acc}

    def val_step(self, batch):
        return {'loss': self._step(batch), 'acc': self.metric.acc}

    def predict_step(self, batch):
        x = batch['img']
        
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        self.preds.append(preds.cpu().numpy())

    def train_epoch_end(self):
        self.print(loss=self.loss.loss, auc=self.metric.auc, acc=self.metric.acc, f1_score=self.metric.f1_score)

    def val_epoch_end(self):
        self.print(loss=self.loss.loss, auc=self.metric.auc, acc=self.metric.acc, f1_score=self.metric.f1_score)

    # save preds to file
    def predict_epoch_end(self):
        pred_name = os.path.join(PRED_PATH, f'{str(env.get("FOLD"))}.csv')
        preds = np.concatenate(self.preds)
        pd.DataFrame({'ImageID': range(1, 28001), 'Label': preds}).to_csv(pred_name, index=None)
        print(f'predict over')

    def _step(self, batch):
        x, y = batch['img'], batch['label']

        logits = self.model(x)
        loss = self.criterion(logits, y)
        probs = F.softmax(logits, dim=1)

        self.loss.update(loss, x.shape[0])
        self.metric.update(probs, y)

        return loss


class MyDataset(Dataset):
    def __init__(self, imgs, labels = None):
        super().__init__()

        self.imgs = imgs.astype('float32')
        self.labels = labels

        self.transform = A.Compose([
            A.Resize(56, 56, interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(0.1310, 0.30854),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.transform(image=self.imgs[idx])['image']
        if self.labels is not None:
            label = self.labels[idx]
            return {'img': img, 'label': label}
        else:
            return {'img': img}

        
class MyLoaderModule(LoaderModule):
    def __init__(self,
                 seed: int,
                 batch_size: int,
                 num_workers: int,
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        train = pd.read_csv('./train.csv')
        self.train_imgs = train.iloc[:, 1:].values.reshape(-1, 28, 28)
        self.train_labels = train.iloc[:, 0].values

        test = pd.read_csv('./test.csv')
        self.test_imgs = test.values.reshape(-1, 28, 28)

        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        self.train_idxs, self.val_idxs = [], []
        for train_idx, val_idx in skf.split(self.train_imgs, self.train_labels):
            self.train_idxs.append(train_idx)
            self.val_idxs.append(val_idx)

    def train_loader(self):
        for train_idx in self.train_idxs:
            dataset = MyDataset(self.train_imgs[train_idx], self.train_labels[train_idx])
            yield DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    def val_loader(self):
        for val_idx in self.val_idxs:
            dataset = MyDataset(self.train_imgs[val_idx], self.train_labels[val_idx])
            yield DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

    def predict_loader(self):
        while True:
            dataset = MyDataset(self.test_imgs)
            yield DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)

if __name__ == '__main__':
    loader_module = MyLoaderModule(**loader_hparams)
    model_module = MyModelModule(model_hparams)
    
    trainer = Trainer(**trainer_hparams, model_module=model_module, loader_module=loader_module)

    write_dict(os.path.join(EXP_PATH, 'hparams.json'), {'model': model_hparams, 'loader': loader_hparams, 'trainer': trainer_hparams})

    trainer.do()
