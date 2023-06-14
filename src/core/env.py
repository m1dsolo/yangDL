import os

from collections import defaultdict
from ..utils.os import rmdir, mkdir

from typing import Literal

__all__ = [
    'set',
    'get',
]

_global_dict = defaultdict(lambda: None, **{
    'LOSS_VAR_NAME': 'loss', # module内loss metric的变量名(用于early stop及进度条打印)
    'LOG_FILE_NAME': 'log.txt', # 会在os.path.join(LOG_PATH, LOG_FILE_NAME)内保存print的结果
    # 'SKIP_INTERACT': False,
    'PRINT_VERBOSE': False,
})

def set(**kwargs):
    global _global_dict
    for key, val in kwargs.items():
        if key == 'EXP_PATH':
            if val[:-1] == '/':
                val = val[:-1]
            rmdir(val)

        _global_dict[key] = val


def get(*keys):
    res = []
    for key in keys:
        if key in ('LOG_PATH', 'METRIC_PATH', 'CKPT_PATH', 'SPLIT_PATH') and key not in _global_dict:
            path = _global_dict['EXP_PATH']
            name = {'LOG_PATH': 'log', 'METRIC_PATH': 'metric', 'CKPT_PATH': 'ckpt', 'SPLIT_PATH': 'split'}[key]
            path = os.path.join(path, name)

            if not os.path.exists(path):
                mkdir(path)

            res.append(path)

        elif key == 'EXP_NAME':
            res.append(os.path.basename(get('EXP_PATH')))

        else:
            res.append(_global_dict.get(key, None))

    return res[0] if len(res) == 1 else res
