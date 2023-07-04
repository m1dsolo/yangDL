import os, torch
from torch import Tensor
from typing import Optional

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from .base_model import BaseModel

__all__ = [
    'SAM',
]

class SAM(BaseModel):
    def __init__(
        self, 
        checkpoint: str,
        pixel_mean: list[float] = [123.675, 116.28, 103.53],
        pixel_std: list[float] = [58.395, 57.12, 57.375]
    ):
        """
        My SAM
        """

        self.resize = ResizeLongestSide(1024)
        sam_type = self._get_sam_type(checkpoint)
        self.sam = sam_model_registry[sam_type](checkpoint=checkpoint)
        self.sam.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.sam.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    def _get_sam_type(self, checkpoint: str):
        gb = os.path.getsize(checkpoint)
        if gb > 2:
            return 'vit_h'
        elif gb > 1:
            return 'vit_l'
        else:
            return 'vit_b'

    def gen_image_embeddings(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            image_embeddings: (B, 256, 64, 64)
        """
        with torch.no_grad():
            x = self.resize.apply_image_torch(x)
            x = self.sam.preprocess(x)
            image_embeddings = self.sam.image_encoder(x)

        return image_embeddings
    
    def __call__(
        self,
        image_embeddings: Tensor,
        original_size: tuple[int, ...],
        points: Optional[Tensor]=None,
        boxes: Optional[Tensor]=None,
        masks: Optional[Tensor]=None,
    ) -> dict[str, Tensor]:
        """
        Args:
            image_embeddings: (1, 256, 64, 64)
            original_size: original image size, (H, W)
            points: (B, N, 2)
            boxes: (B, 4)
            masks: (B, 1, H, W)
        """
        if points is not None:
            points = self.resize.apply_coords_torch(points)
        if boxes is not None:
            boxes = self.resize.apply_boxes_torch(boxes, original_size)

        sparse, dense = self.sam.prompt_encoder(points=points, boxes=boxes, masks=masks)
        low_res_logits, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )

        input_size = ResizeLongestSide.get_preprocess_shape(original_size[0], original_size[1], 1024)
        logits = self.sam.postprocess_masks(low_res_logits, input_size, original_size)

        # (B, 1, 512, 512), (B, 1)
        return {'logits': logits, 'iou_predictions': iou_predictions}

    def cuda(self, device = None):
        self.sam = self.sam.cuda(device)
        return self

    def to(self, device = None):
        self.sam = self.sam.to(device)
        return self

    def train(self):
        self.sam.train()

    def eval(self):
        self.sam.eval()

    @property
    def image_encoder(self):
        return self.sam.image_encoder

    @property
    def prompt_encoder(self):
        return self.sam.prompt_encoder

    @property
    def mask_decoder(self):
        return self.sam.mask_decoder
