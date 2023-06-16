import torch
from torch import Tensor

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

from typing import Optional, Sequence

from .metric import Metric
from ..utils.dl import bincount


__all__ = [
    'ConfusionMatrix',
]


class ConfusionMatrix(Metric):
    def __init__(
        self, 
        num_classes: int,
        thresh: Optional[float | str] = None, 
        ignore_idxs: Sequence[int] = [],
        eps: float = 1e-7,

        prefix: Optional[str] = None,
        freq: tuple[str, int] = ('epoch', 1),
    ):
        """
        Confusion Matrix

        Args:
            num_classes: class num, each label should be in the range [0, C]
            thresh: (only useful in binary classification), confusion matrix thresh, if use float will not need to save probs and labels
                - None: for num_classes > 2
                - float: as matrix thresh value
                - 'f1_score': will use thresh that has best f1_score
                - 'roc': will use thresh that has best tpr - fpr
            ignore_idxs: use to ignore idx
            eps: prevent division by 0
            prefix, freq: see Metric class
        """

        """ 二分类混淆矩阵 
              预测
              0  1
         真 0 TN FP
         实 1 FN TP
        """

        """ 多分类混淆矩阵(以A类positive为例)
               预 测
              A  B  C
         真 A TP FN FN
            B FP TN TN
         实 C FP TN TN
        """
        super().__init__(prefix, freq)

        if num_classes > 2:
            assert thresh is None, f'num_classes > 2, thresh must equal to None!'
        elif thresh is None:
            thresh = 0.5

        self.num_classes = num_classes
        self._thresh = thresh
        self.add_tensor('_matrix', torch.zeros((num_classes, num_classes), dtype=torch.long))
        self.eps = eps

        self.add_tensor('probs', torch.tensor([], dtype=torch.float)) # num_classes == 2: (B,), else: (B, C)
        self.add_tensor('labels', torch.tensor([], dtype=torch.long)) # (B,)

    def _calc_matrix(
        self, 
        probs: Tensor, 
        labels: Tensor, 
    ) -> Tensor:

        preds = self._calc_preds(probs)
        bins = bincount(labels * self.num_classes + preds, minl=self.num_classes ** 2)
        return bins.reshape(self.num_classes, self.num_classes)
    
    def _calc_preds(
        self,
        probs: Tensor,
    ) -> Tensor:

        if self.num_classes == 2:
            return (probs > self.thresh).long()
        else:
            return probs.argmax(dim=1).long()

    @torch.no_grad()
    def update(
        self,
        probs: Tensor,
        labels: Tensor,
    ):
        """
        use probs and labels to update metric

        Args:
            probs: (B, C), value in [0, 1], always == F.softmax(logits, dim=1)
            labels: (B,), value in [0, C - 1]
        """
        assert len(probs) == len(labels), f'probs.shape={probs.shape}, labels.shape={labels.shape}'

        probs, labels = probs.detach(), labels.detach()

        if not isinstance(self._thresh, str):
            self._matrix += self._calc_matrix(probs, labels)
        if self.num_classes == 2:
            probs = probs[:, 1]

        self.probs = torch.cat([self.probs, probs], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)

    def ravel(self, idx: Optional[int] = None):
        """
        confusion matrix convert to (tn, fp, fn, tp)

        Args:
            idx: positive class idx
        """
        tp = self.matrix[idx, idx]
        fp = self.matrix[:, idx].sum() - tp
        fn = self.matrix[idx, :].sum() - tp
        tn = self.matrix.sum() - tp - fp - fn
        return tn, fp, fn, tp

    @property
    def matrix(self):
        if isinstance(self._thresh, str) and self._matrix.sum() != len(self.probs):
            self._matrix = self._calc_matrix(self.probs, self.labels)
        return self._matrix

    @property
    def acc(self):
        return self.matrix.diagonal().sum() / (self.matrix.sum() + self.eps)

    @property
    def pos_acc(self):
        assert self.num_classes == 2, f'num_classes={self.num_classes}'

        return self.matrix[1, 1] / (self.matrix[1, :].sum() + self.eps)

    @property
    def neg_acc(self):
        assert self.num_classes == 2, f'num_classes={self.num_classes}'

        return self.matrix[0, 0] / (self.matrix[0, :].sum() + self.eps)

    @property
    def precision(self):
        res = []
        for idx in range(self.num_classes == 2, self.num_classes):
            tn, fp, fn, tp = self.ravel(idx)
            res.append(tp / (tp + fp + self.eps))
        return torch.stack(res, dim=0).mean()
    
    @property
    def recall(self):
        res = []
        for idx in range(self.num_classes == 2, self.num_classes):
            tn, fp, fn, tp = self.ravel(idx)
            res.append(tp / (tp + fn + self.eps))
        return torch.stack(res, dim=0).mean()

    @property
    def sensitivity(self):
        return self.recall

    @property
    def specificity(self):
        res = []
        for idx in range(self.num_classes == 2, self.num_classes):
            tn, fp, fn, tp = self.ravel(idx)
            res.append(tn / (tn + fp + self.eps))
        return torch.stack(res, dim=0).mean()

    @property
    def f1_score(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall + self.eps)

    @property
    def auc(self):
        res = -1.
        if self.labels.sum().item() not in (0, len(self.labels)):
            res = roc_auc_score(self.labels.cpu().numpy(), self.probs.cpu().numpy(), multi_class='ovr')
        return torch.tensor(res, dtype=torch.float)

    @property
    def ap(self):
        assert self.num_classes == 2, f'num_classes={self.num_classes}'

        res = -1.
        if self.labels.sum().item() not in (0, len(self.labels)):
            res = average_precision_score(self.labels.cpu().numpy(), self.probs.cpu().numpy())
        return torch.tensor(res, dtype=torch.float)
    
    @property
    def thresh(self):
        assert self.num_classes == 2, f'num_classes={self.num_classes}'

        if self._thresh is None:
            thresh = -1.

        if not isinstance(self._thresh, str):
            thresh = self._thresh
        else:
            labels, probs = self.labels.cpu().numpy(), self.probs.cpu().numpy()
            if labels.sum().item() in (0, len(labels)):
                thresh = 0.5
            else:
                if self._thresh == 'f1_score':
                    precisions, recalls, threshs = precision_recall_curve(labels, probs)
                    f1s = 2 * precisions * recalls / (precisions + recalls + self.eps)
                    thresh = threshs[f1s.argmax(axis=0)]
                elif self._thresh == 'roc':
                    fpr, tpr, threshs = roc_curve(labels, probs, pos_label=1)
                    thresh = threshs[(tpr - fpr).argmax(axis=0)]

        return torch.tensor(thresh, dtype=torch.float)
