import random
from functools import reduce
from collections import Counter, defaultdict
from queue import PriorityQueue

from typing import Sequence, Optional

# __all__ = [
    # 'list_reshape',
    # 'scale',
    # 'match_substr',
    # 'dict_val_to_list',
    # 'merge_dict',
    # 'sort_dict',
    # ''
# ]

# l=[1, 2, 3, 4, 5, 6], shape=(2, 3) --> res=[[1, 2, 3], [4, 5, 6]]
def list_reshape(lst: list, shape: Sequence) -> list:
    assert len(lst) == reduce(lambda a, b: a * b, shape)

    if len(shape) == 1:
        return lst
    
    res = []
    step = reduce(lambda a, b: a * b, shape[1:])
    for i in range(0, len(lst), step):
        res.append(list_reshape(lst[i:i + step], shape[1:]))
    return res

# l=[1920, 1080], k=640, upper=True --> res=[640, 360]
# l=[1920, 1080], k=2160, upper=False --> res=[3840, 2160]
def scale(lst: Sequence, k: int, dtype=int, upper: bool = True) -> list:
    k = k / (max(lst) if upper else min(lst))
    return list(map(lambda a: dtype(a * k), lst))

# s="abcdef", subs=["abc", "xyz"] --> res="abc"
def match_substr(s: str, subs: Sequence[str]) -> Optional[str]:
    for sub in subs:
        if sub in s:
            return sub
    return None

# {'a': 1, 'b': 2} --> {'a': [1], 'b': [2]}
def dict_val_to_list(d: Optional[dict]):
    if d is None:
        return d
    return {key: list(val) if isinstance(val, Sequence) else list([val]) for key, val in d.items()}

# d1={'a': 1, 'b': 2}, d2={'b': 3, 'c': 4} --> {'a': [1], 'b': [2, 3], 'c': 4}
def merge_dict(d1: Optional[dict], d2: Optional[dict]) -> Optional[dict]:
    if d1 is not None:
        assert isinstance(d1, dict)
    if d2 is not None:
        assert isinstance(d2, dict)
    if d1 is None:
        return d2
    if d2 is None:
        return d1
    d1 = dict_val_to_list(d1)
    d2 = dict_val_to_list(d2)

    for key in d2:
        if key in d1:
            d1[key].extend(d2[key])
        else:
            d1[key] = d2[key][:]

    return d1

# sort dict by key
def sort_dict(d: dict, keys: Optional[Sequence[str]] = None, reverse: bool = False):
    if keys is None:
        return dict(sorted(d.items(), key=lambda kv: kv[0], reverse=reverse))
    else:
        if reverse:
            keys = reversed(keys)
        return {key: d[key] for key in keys}

def calc_weights(labels: Sequence):
    weights = sort_dict(Counter(labels)).values()
    weights = list(map(lambda w: 1 / w, weights))
    s = sum(weights)
    return list(map(lambda w: w / s, weights))

def shuffle(*args, seed=None) -> list:
    l = len(args[0])
    for lst in args:
        assert len(lst) == l

    if seed is not None:
        random.seed(seed)
    idxs = list(range(l))
    random.shuffle(idxs)

    if len(args) == 1:
        return [lst[idx] for idx in idxs]
    else:
        return [[lst[idx] for idx in idxs] for lst in args]

# unique by the first arg of args
def unique(*args) -> list[list]:
    st = set()
    res = [[] for _ in range(len(args))]
    for row in zip(*args):
        if row[0] not in st:
            for i, val in enumerate(row):
                res[i].append(val)
            st.add(row[0])
    return res if len(res) > 1 else res[0]

# (num=15, k=6) --> [3, 3, 3, 2, 2, 2]
def split_num(num: int, k: int):
    sub = num // k
    return [sub + bool(i < num - sub * k) for i in range(k)]

def split_sequence(*args: Sequence, k: int):
    for l in args:
        l = list(l)
        res = []
        i = 0
        for num in split_num(len(l), k):
            res.append(l[i:i + num])
            i += num
        yield res

# l=[1, 2, 3, 4, 5], gids=[0, 1, 0, 2, 3] --> [[1, 3], [2], [4], [5]]
def group_list(l: Sequence, gids: Sequence) -> list[list]:
    d = defaultdict(list)
    for val, gid in zip(l, gids):
        d[gid].append(val)
    return list(d.values())

# l=[1, 2, 3, 4, 5], k=3 --> [[1, 2], [3, 4], [5]]
# l=[1, 2, 3, 4, 5], gids=[0, 1, 0, 2, 3], k=3 --> [[1, 3], [2, 4], [5]]
def split_list(l: Sequence, 
               k: int, 
               gids: Optional[Sequence] = None
               ) -> Optional[list[list]]:
    if gids is None:
        res = []
        i = 0
        for num in split_num(len(l), k):
            res.append(l[i:i + num])
            i += num
        return res
    else:
        assert len(l) == len(gids)
        
        pq = PriorityQueue()
        nums = split_num(len(l), k)
        for i, num in enumerate(nums):
            pq.put((-num, i))

        res = [[] for _ in range(k)]
        ls = group_list(l, gids)
        ls.sort(key=lambda l: len(l), reverse=True)
        for l in ls:
            idx = pq.get()[1]
            if len(l) > nums[idx]:
                return None
            nums[idx] -= len(l)
            res[idx].extend(l)
            pq.put((-nums[idx], idx))

        return res

# def read_metrics(stage: str, 
                 # metric_path: str, 
                 # names: Optional[str] = None
                 # ) -> dict:
    # res = read_json(os.path.join(metric_path, 'early_stop.json'))
