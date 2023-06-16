import os, torch, copy
import numpy as np
from torch import nn, Tensor

from typing import Optional, Sequence

__all__ = [
    'set_seed',
    'bincount',
    'ModelExtractor',
    'get_model_devices',
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

# 存在问题: output为tuple时
class ModelExtractor():
    def __init__(
        self, 
        model: nn.Module, 
        module_names: Optional[Sequence[str]] = None, 
        reshape_transforms=None
    ):
        """
        use to extract model activations and gradients

        Args:
            model: model to extract
            module_names: get module's activations and gradients which in module_names
                - None: register all module in model.named_modules()
                - Sequence[str]: register all module in module_names
            reshape_transforms: TODO

        Examples:
        """
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.reshape_transforms = reshape_transforms
        self.activations = {}
        self.gradients = {}

        named_modules = dict(self.model.named_modules())
        for module_name in module_names or named_modules.keys():
            module = named_modules[module_name]
            module.register_forward_hook(self.forward_hook(module_name))
            if hasattr(module, 'register_full_backward_hook'):
                module.register_full_backward_hook(self.backward_hook(module_name))

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
