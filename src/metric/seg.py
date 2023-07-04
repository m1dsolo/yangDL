import torch
from torch import Tensor

from typing import Optional, Sequence

from .metric import Metric
from .confusion_matrix import ConfusionMatrix

from ..utils.dl import bincount


__all__ = [
    'SegMetric',
]

class SegMetric(Metric):
    def __init__(
        self, 
        num_classes: int,
        properties: Optional[Sequence[str]] = None,
        thresh: Optional[float | str] = None, 
        ignore_idxs: Sequence[int] = [],
        eps: float = 1e-7, 

        prefix: Optional[str] = None,
        freq: tuple[str, int] = ('epoch', 1),
    ):
        """
            segmentation metric

            Args:
                properties: properties which will auto log to tensorboard
                    - None: all properties
                    - ['auc', 'acc']: will only log auc and acc
                num_classes, thresh, ignore_idxs, eps: see ConfusionMatrix
                prefix, freq: see Metric class
        """

        self.num_classes = num_classes
        if properties is None:
            if num_classes == 2:
                self.properties = ['acc', 'pos_acc', 'neg_acc', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score', 'dice', 'iou', 'thresh']
            else:
                self.properties = ['acc', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score', 'dice', 'iou']
        else:
            self.properties = properties

        super().__init__(prefix, freq)

        self.cm = ConfusionMatrix(num_classes=num_classes, thresh=thresh, ignore_idxs=ignore_idxs, eps=eps, save_probs=False)

    @torch.no_grad()
    def update(
        self,
        probs: Tensor,
        labels: Tensor,
    ):
        """
        use probs and labels to update metric

        Args:
            if num_classes == 2:
                probs.shape == (B, H, W), value in [0, 1]
                labels.shape == (B, H, W), value in (0, 1)
            else:
                probs.shape == (B, C, H, W), value in [0, 1]
                labels.shape == (B, H, W), value in (0, ..., C - 1)
        """
        if self.num_classes == 2:
            assert probs.shape == labels.shape and len(probs.shape) == 3, f'probs.shape={probs.shape}, labels.shape={labels.shape}'
            probs = probs.flatten(0, 2)
        else:
            assert len(probs.shape) == 4 and len(labels.shape) == 3, f'probs.shape={probs.shape}, labels.shape={labels.shape}'
            probs = probs.permute(0, 2, 3, 1).flatten(0, 2)
        labels = labels.flatten(0, 2)

        self.cm.update(probs, labels)

    @property
    def matrix(self):
        return self.cm.matrix

    @property
    def acc(self):
        return self.cm.acc

    @property
    def pos_acc(self):
        return self.cm.pos_acc

    @property
    def neg_acc(self):
        return self.cm.neg_acc

    @property
    def precision(self):
        return self.cm.precision
    
    @property
    def recall(self):
        return self.cm.recall

    @property
    def sensitivity(self):
        return self.cm.sensitivity

    @property
    def specificity(self):
        return self.cm.specificity

    @property
    def f1_score(self):
        return self.cm.f1_score

    @property
    def dice(self):
        return self.cm.dice

    @property
    def iou(self):
        return self.cm.iou

    @property
    def thresh(self):
        return self.cm.thresh
