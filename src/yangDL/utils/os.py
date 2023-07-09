import os, shutil

from typing import Optional, Union

__all__ = [
    'get_file_names',
    'get_path_file_names'
    'mkdir',
    'rmdir',
    'mv',
    'cp',
    'rm',
]

def get_file_names(
    path: str, 
    suffix: Optional[str] = None, 
    type: Optional[str] = None, 
    with_dir: bool = False, 
    with_suffix: bool = False
) -> list[str]:
    """
    get file names, the file names will sorted, if file names are all number, they will sorted by numerically rather than lexicographically

    Example:
    dir structure:
        path:
            aaa.png
            bbb.png

    >>> print(get_path_file_names(path, '.png'))
    ['aaa', 'bbb']
    >>> print(get_path_file_names(path, '.png', with_dir=True))
    ['path/aaa', 'path/bbb']
    >>> print(get_path_file_names(path, '.png', with_suffix=True))
    ['aaa.png', 'bbb.png']

    Args:
        path: the path to get file names
        suffix: if suffix is not None: only file_name[-len(suffix):] == suffix will return, and return file name will not with suffix
        type: if type is dir: will only return dir name
        with_dir: all return file name will with its path
        with_suffix: all
    """

    file_names = os.listdir(path)
    all_num = True
    for file_name in file_names:
        if not file_name.split('.')[0].isdigit():
            all_num = False
            break

    if type == 'dir':
        file_names = list(filter(lambda file_name: os.path.isdir(os.path.join(path, file_name)), file_names))
    file_names = sorted(file_names, key=lambda x: int(x.split('.')[0])) if all_num else sorted(file_names)

    if suffix:
        file_names = [(file_name if with_suffix else file_name[:-len(suffix)]) for file_name in file_names if file_name[-len(suffix):] == suffix]

    if with_dir:
        file_names = [os.path.join(path, file_name) for file_name in file_names]

    return file_names

def get_path_file_names(
    path: str, 
    with_path: bool = False
) -> list[str]:
    res = []
    abs_path = os.path.abspath(path)
    for root, _, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            if not with_path:
                path = path[len(abs_path) + 1:]
            res.append(path)
    return res

def mkdir(*args: str) -> None:
    for dir_name in args:
        os.makedirs(dir_name, exist_ok=True)


def rmdir(*args: str) -> None:
    for dir_name in args:
        # if os.path.exists(dir_name):
        shutil.rmtree(dir_name, ignore_errors=True)

def mv(src: str, dst: str) -> None:
    shutil.move(src, dst)

def cp(src: str, dst: str) -> None:
    if os.path.isdir(src):
        rmdir(dst)
        shutil.copytree(src, dst)
    else:
        mkdir(os.path.dirname(dst))
        shutil.copy(src, dst)

def rm(*args: str) -> None:
    for file_name in args:
        os.remove(file_name)
