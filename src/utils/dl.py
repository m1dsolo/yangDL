import os, torch, copy
import numpy as np
from torch import nn

from typing import Optional, Sequence

__all__ = [
    'set_seed',
    'bincount',
    'ModelExtractor',
    'get_model_devices',
    'mask2box',
]

def set_seed(seed: Optional[int] = None):
    if seed is None:
        return
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# x=[1, 1, 0, 2, 2] --> res=[1, 2, 2]
def bincount(x: torch.Tensor, minl: Optional[int] = None):
    if minl is None:
        minl = x.max() + 1
    res = torch.empty(minl, device=x.device, dtype=torch.long)
    for i in range(minl):
        res[i] = (x == i).sum()
    return res

# 存在问题: output为tuple时(DSMIL output)
class ModelExtractor():
    def __init__(self, model: nn.Module, names: Sequence = None, reshape_transforms=None):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.reshape_transforms = reshape_transforms
        self.activations = {}
        self.gradients = {}
        named_modules = dict(self.model.named_modules())
        if names is None:
            names = list(named_modules.keys())
        for name in names:
            module = named_modules[name]
            module.register_forward_hook(self.forward_hook(name))
            if hasattr(module, 'register_full_backward_hook'):
                module.register_full_backward_hook(self.backward_hook(name))

    def forward_hook(self, name):
        def fn(module, input, output):
            activation = output
            if self.reshape_transforms is not None:
                activation = self.reshape_transforms(activation)
            self.activations[name] = activation.cpu().detach()
        return fn

    def backward_hook(self, name):
        def fn(module, grad_input, grad_output):
            gradient = grad_output[0]
            if self.reshape_transforms is not None:
                gradient = self.reshape_transforms(gradient)
            self.gradients[name] = gradient.cpu().detach()
        return fn

    def __call__(self, x):
        return self.model(x)

def get_model_devices(model):
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    return devices

# mask (H, W) (used for SAM)
def mask2box(mask: torch.Tensor, padding: int) -> torch.Tensor:
    y_indices, x_indices = torch.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    x_min = max(0, x_min - np.random.randint(0, padding))
    x_max = min(mask.shape[1] - 1, x_max + np.random.randint(0, padding))
    y_min = max(0, y_min - np.random.randint(0, padding))
    y_max = min(mask.shape[0] - 1, y_max + np.random.randint(0, padding))

    return torch.tensor([x_min, y_min, x_max, y_max])
