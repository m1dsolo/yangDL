import os, torch
from torch import Tensor
from typing import Optional

from segment_anything.modeling import Sam
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
        My SAM, freeze image encoder
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
        image: Optional[Tensor]=None,
        original_size: Optional[tuple[int, ...]]=None,
        image_embeddings: Optional[Tensor]=None,
        points: Optional[Tensor]=None,
        boxes: Optional[Tensor]=None,
        masks: Optional[Tensor]=None,
    ) -> dict[str, Tensor]:
        """
        do one image

        Args:
            image: (3, H, W)
            original_size: original image size, (H, W)
            image_embeddings: (256, 64, 64)
            points: (B, N, 2)
            boxes: (B, 4)
            masks: (B, H, W)

        Returns:
        """
        if original_size is None:
            assert image is not None
            original_size = image.shape[-2:]
        if image_embeddings is None:
            assert image is not None
            with torch.no_grad():
                image_embeddings = self.gen_image_embeddings(image[None])[0]
        if points is not None:
            points = self.resize.apply_coords_torch(points)
        if boxes is not None:
            boxes = self.resize.apply_boxes_torch(boxes, original_size)
        if masks is not None:
            masks = masks[:, None, ...]

        sparse, dense = self.sam.prompt_encoder(points=points, boxes=boxes, masks=masks)
        low_res_logits, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings[None],
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )

        input_size = ResizeLongestSide.get_preprocess_shape(original_size[0], original_size[1], 1024)
        logits = self.sam.postprocess_masks(low_res_logits, input_size, original_size)

        # (B, 512, 512), (B)
        return {'logits': logits[:, 0, ...], 'iou_predictions': iou_predictions[:, 0]}

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
