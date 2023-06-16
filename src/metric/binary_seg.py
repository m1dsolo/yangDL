import torch
from torch import Tensor
import segmentation_models_pytorch as smp
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

from typing import Optional, Union, Sequence, Literal

from .metric import Metric
from ..utils.dl import bincount

class BinarySegMetric(Metric):
    def __init__(self, thresh: float = 0.5, eps: float = 1e-7, devices: Optional[Sequence] = None, prefix: Optional[str] = None, on: Literal['step', 'epoch'] = 'epoch', freq: int = 1):
        super().__init__(devices, prefix, on, freq)

        self._matrix = torch.zeros((2, 2), dtype=torch.long)
        self.thresh = thresh
        self.eps = eps

    def _calc_matrix(self, probs: Tensor, labels: Tensor, thresh: Union[int, float]):
        x = (labels * 2 + (probs > thresh)).to(torch.long)
        bins = bincount(x, minl=4)

        return bins.reshape(2, 2)

    @torch.no_grad()
    # (B, 1, ...) 
    def update(self, probs: Tensor, target: Tensor):
        probs, target = probs.squeeze(), target.squeeze()
        assert probs.shape == target.shape
        probs, target = probs.reshape(-1), target.reshape(-1)
        probs, target = probs.detach().cpu(), target.detach().cpu()

        self._matrix += self._calc_matrix(probs, target, self.thresh)

    def reset(self):
        self._matrix = torch.zeros((2, 2), dtype=torch.long)

        self.to(self.devices)

    @property
    def dice(self):
        return (2 * self.matrix[1, 1]) / (self.matrix[0, 1] + 2 * self.matrix[1, 1] + self.matrix[1, 0] + self.eps)

    @property
    def iou(self):
        return self.matrix[1, 1] / (self.matrix[1, 1] + self.matrix[0, 1] + self.matrix[1, 0] + self.eps)

    @property
    def matrix(self):
        return self._matrix

    @property
    def acc(self):
        return (self.matrix[0, 0] + self.matrix[1, 1]) / (self.matrix.sum() + self.eps)

    @property
    def pos_acc(self):
        return self.matrix[1, 1] / (self.matrix[1, :].sum() + self.eps)

    @property
    def neg_acc(self):
        return self.matrix[0, 0] / (self.matrix[0, :].sum() + self.eps)

    @property
    def precision(self):
        return self.matrix[1, 1] / (self.matrix[:, 1].sum() + self.eps)
    
    @property
    def recall(self):
        return self.matrix[1, 1] / (self.matrix[1, :].sum() + self.eps)

    @property
    def sensitivity(self):
        return self.matrix[1, 1] / (self.matrix[1, :].sum() + self.eps)

    @property
    def specificity(self):
        return self.matrix[0, 0] / (self.matrix[0, :].sum() + self.eps)

    @property
    def f1_score(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall + self.eps)
