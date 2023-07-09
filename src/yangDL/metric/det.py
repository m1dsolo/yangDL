import torch
from torch import Tensor
import numpy as np

from typing import Optional, Sequence

from .metric import Metric
from .confusion_matrix import ConfusionMatrix

from ..utils.dl import bincount


__all__ = [
    'DetMetric',
]

class DetMetric(Metric):
    def __init__(
        self,
        num_classes: int,
        box_format: str = 'xyxy',
        iou_type: str = 'bbox',
        eps: float = 1e-7,

        prefix: Optional[str] = None,
        freq: tuple[str, int] = ('epoch', 1),
    ):
        """
            detection metric.
            will use N == num_pred_boxes, M == num_gt_boxes, to simplified annotations.

            Args:
                num_classes: class num, each label should be in the range [0, C).
                box_format: 'xyxy' or 'xywh'.
                iou_type: 'bbox' or 'seg'.
                eps: prevent division by 0
                prefix, freq: see Metric class
        """

        self.num_classes = num_classes
        self.box_format = box_format
        self.iou_type = iou_type
        self.eps = eps
        self.preds: list[tuple[Tensor, Tensor, Tensor]] = [] # list['dets': (N, 4) or (N, H, W), 'scores': (N,), 'labels': (N,)]
        self.labels: list[tuple[Tensor, Tensor]] = [] # list['gts': (M, 4) or (M, H, W), 'labels': (M,)]

    def update(
        self,
        preds: Sequence[dict[str, Tensor]],
        labels: Sequence[dict[str, Tensor]],
    ):
        """
        use probs and labels to update metric

        Args:
            preds: list of dict each containing (str, Tensor) pairs:
                len(preds) == batch_size

                - 'dets': 
                    - if self.iou_type == 'bbox': (N, 4), in format self.box_format
                    - if self.iou_type == 'seg': (N, H, W), value in (0, 1)
                - 'scores': (N,), detection scores
                - 'labels': (N,), classification result
            labels: list of dict each containing (str, Tensor) pairs:
                len(labels) == batch_size

                - 'gts':
                    - if self.iou_type == 'bbox': (M, 4), in format self.box_format
                    - if self.iou_type == 'seg': (N, H, W), value in (0, 1)
                - 'labels': (M,), classification gt
        """
        assert len(preds) == len(labels)

        self.preds.extend(preds)
        self.labels.extend(labels)

    def calc_ap(self, thresh: float | Sequence[float]):
        return self._compute_aps(thresh)

    @property
    def ap(self):
        return self._compute_aps(np.arange(0.5, 1., 0.05))

    @property
    def ap50(self):
        return self._compute_aps(0.5)

    @property
    def ap55(self):
        return self._compute_aps(0.55)

    @property
    def ap60(self):
        return self._compute_aps(0.55)

    def _compute_aps(self, thresh: float | Sequence[float]):
        """
        compute aps

        Returns:
            aps: (len(thresh[Sequence]))
        """
        if isinstance(thresh, float):
            thresh = [thresh]

        ious = [self._calc_ious(dets, gts) for dets, gts in zip(self.preds['dets'], self.labels['gts'])]
        aps = torch.zeros((len(thresh), len(self.num_classes)), dtype=torch.long)
        for c in range(len(self.num_classes)):
            c_scores, c_ious, fp, fn = [], [], 0, 0
            for img_id, (pred, label) in enumerate(zip(self.preds, self.labels)):
                pred_idx, label_idx = pred['labels'] == c, label['labels'] == c
                dets = self.preds['dets'][img_id][pred_idx]
                gts = self.labels['gts'][img_id][label_idx]
                scores = self.preds['scores'][img_id][pred_idx]

                img_scores, img_ious, img_fp, img_fn = self._compute_one(dets, gts, scores, ious[img_id])
                c_scores.extend(img_scores)
                c_ious.extend(img_ious)
                fp += img_fp
                fn += img_fn

            c_scores = torch.cat(c_scores, dim=0)
            c_ious = torch.cat(c_ious, dim=0)
            c_scores, idx = c_scores.sort(descending=True)
            c_ious = c_ious[idx]
            for i, th in enumerate(thresh):
                aps[i][c] = self._calc_ap(c_ious > th, c_scores, fp, fn)

    def _calc_ap(self, preds: Tensor, labels: Tensor, fp: int, fn: int):

    def _compute_one(
        self,
        dets: Tensor,
        gts: Tensor,
        scores: Tensor,
        ious: Tensor
    ) -> tuple[list, list, int]:
        """
        compute one img's scores and ious

        Args:
            dets: (N, 4) or (N, H, W), one img's dets
            gts: (M, 4) or (M, H, W), one img's gts
            scores: (N,), one img's scores
            ious: (N, M), one img's ious
        Returns:
            res_scores: (A,)
            res_ious: (A,)
            fp: num of fp
            fn: num of fn
        """
        match_ious, match_ids = ious.max(ious, dim=1)
        vis = torch.zeros(len(gts), dtype=bool)
        _, idxs = scores.sort(descending=True)

        res_scores = []
        res_ious = []
        fp = 0
        for idx in idxs:
            score, match_iou, gt_id = scores[idx], match_ious[idx], match_ids[idx]
            if vis[gt_id]:
                fp += 1
            else:
                vis[gt_id] = True
                res_scores.append(score)
                res_ious.append(match_iou)

        fn = (vis == 0).sum()
        return res_scores, res_ious, fp, fn


    def _calc_ious(
        self,
        dets: Tensor,
        gts: Tensor,
    ) -> Tensor:
        """
        calulate iou between det and gt

        Args:
            dets: (N, 4) or (N, H, W)
            gts: (M, 4) or (M, H, W)

        Returns:
            ious: (N, M)
        """

        if self.iou_type == 'bbox':
            pass
        else:
            dets = dets.flatten(1, 2)
            gts = gts.flatten(1, 2)

            inter = (dets @ gts.T)
            area1 = torch.sum(dets, dim=1).reshape(1, -1)
            area2 = torch.sum(gts, dim=1).reshape(1, -1)
            union = area1.T + area2 - inter

            iou = inter / (union + self.eps)

        return torch.tensor(iou, dtype=torch.float)
