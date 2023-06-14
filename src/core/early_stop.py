import numpy as np

__all__ = [
    'EarlyStop',
]

class EarlyStop():
    def __init__(self, patience=20, min_stop_epoch=50, max_stop_epoch=200, eps: float = 1e-5):
        self.patience = patience
        self.min_stop_epoch = min_stop_epoch
        self.max_stop_epoch = max_stop_epoch
        self.eps = eps

        self.init()

        self.res = {'stop_epoch': [], 'best_epoch': []}

    def init(self):
        self.counter = 0
        self.min_loss = np.Inf

        self.stop_epoch = 0
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, loss, epoch):
        if loss < self.min_loss - self.eps:
            self.min_loss = loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1

        if self.counter > self.patience and epoch >= self.min_stop_epoch or epoch == self.max_stop_epoch:
            self.stop_epoch = epoch
            self.early_stop = True

            self.res['stop_epoch'].append(epoch)
            self.res['best_epoch'].append(self.best_epoch)

    def to_dict(self):
        return self.res
