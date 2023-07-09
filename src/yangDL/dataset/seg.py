import os, torch
import numpy as np
from PIL import Image

from collections import Counter
from typing import Optional

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms.functional import to_tensor

from yangDL.utils.os import get_file_names

class SegDataset(Dataset):
    """
    format:

    dataset_file_name:
        img:
            file_name_1:
                1.png
                2.png
            file_name_2:
                1.png
        gt:
            file_name_1:
                1.png
                2.png
            file_name_2:
                1.png
    """
    def __init__(self, 
                 dataset_path: str, 
                 img_path: Optional[str] = None,
                 gt_path: Optional[str] = None,
                 file_names: Optional[list[str]] = None,
                 transform: Optional[list | Compose] = None,
                 ):
        super().__init__()

        if img_path is None: img_path = os.path.join(dataset_path, 'img')
        if gt_path is None: gt_path = os.path.join(dataset_path, 'gt')
        if file_names is None: file_names = get_file_names(gt_path, type='dir')

        self.img_file_names = []
        self.gt_file_names = []
        for file_name in file_names:
            for png_name in get_file_names(os.path.join(gt_path, file_name), '.png'):
                self.img_file_names.append(os.path.join(img_path, file_name, png_name + '.png'))
                self.gt_file_names.append(os.path.join(gt_path, file_name, png_name + '.png'))

        self.transform = transform
        if isinstance(self.transform, list):
            self.transform = Compose(self.transform)

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self, idx):
        img = to_tensor(Image.open(self.img_file_names[idx]).convert('RGB'))
        gt = torch.from_numpy(np.array(Image.open(self.gt_file_names[idx]).convert('L')))
        gt[gt > 0] = 1

        if self.transform is not None:
            img = self.transform(img)
        return img, gt
