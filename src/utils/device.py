from time import sleep
import numpy as np

from nvitop import Device

__all__ = [
    'wait_gpu',
    'get_gpu_mems',
    'get_max_gpu',
]

def wait_gpu(need: int, sleep_secs: int = 1) -> int:
    """
    wait the first gpu which has mem >= need
    Returns:
        idx: ok gpu idx
    """
    devices = Device.all()
    while True:
        i, mem = get_max_gpu()
        if mem >= need:
            return i
        sleep(sleep_secs)

def get_gpu_mems() -> list[float]:
    return list(map(lambda device: device.memory_free() / (1 << 20), Device.all()))

def get_max_gpu() -> tuple[int, float]:
    """
    Returns:
        idx: gpu idx which has max memory
        mem: max memory
    """
    mems = np.array(get_gpu_mems())
    idx = mems.argmax()

    return idx, mems[idx]
