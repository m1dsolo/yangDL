import rich
from rich.live import Live

from typing import Optional

from ..core import env

class WithNone():
    def __enter__(self):
        pass
    def __exit__(self, err_type, err_val, err_pos):
        pass
