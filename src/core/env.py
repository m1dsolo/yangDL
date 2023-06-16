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
        _global_dict[key] = val

        if key == 'EXP_PATH':
            for path_key, path_name in {'LOG_PATH': 'log', 'METRIC_PATH': 'metric', 'CKPT_PATH': 'ckpt'}.items():
                _global_dict[path_key] = os.path.join(key, path_name)
                mkdir(_global_dict[path_key])

            EXP_NAME = os.path.basename(val if val[:-1] != '/' else val[:-1])
            _global_dict['EXP_NAME'] = EXP_NAME


def get(*keys):
    res = [_global_dict.get(key, None) for key in keys]

    return res[0] if len(res) == 1 else res
