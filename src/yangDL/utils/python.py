import inspect, importlib, re

from typing import Callable, Any

def get_properties(obj) -> dict:
    return {key: obj.__getattribute__(key) for key, val in vars(obj.__class__).items() if isinstance(val, property)}

# num=0 get cur_frame, num=1 get back_frame and so on
def get_func_name(num: int = 0) -> str:
    frame = inspect.currentframe()
    for _ in range(num + 1):
        frame = frame.f_back
    return frame.f_code.co_name

def get_class_name_from_method(f: Callable):
    return f.__qualname__.split('.')[0]

def get_class_from_method(f: Callable):
    cls_name = get_class_name_from_method(f)
    return getattr(importlib.import_module(f.__module__), cls_name)

# f=a.func
def method_is_overrided_in_subclass(f: Callable):
    cls = get_class_from_method(f)
    for parent in cls.__mro__[1:]:
        if f.__name__ in parent.__dict__:
            return True
    return False

# num=0: cur frame, num=1: last frame
def get_params(num: int = 0):
    frame = inspect.currentframe()
    for _ in range(num + 1):
        frame = frame.f_back
    return {key: val for key, val in frame.f_locals.items() if key not in ('self', '__class__')}

# def dict_get(d: dict, *keys: Any):
    # return (d[key] for key in keys)

# 去除方括号及其内的内容
def clear_rich(s: str):
    return re.sub(r'(\[.*?\])', '', s)
