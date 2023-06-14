import os, json, h5py
import numpy as np
import pandas as pd

from typing import Optional, Sequence, Any

from .os import mkdir

__all__ = [
    'write_dict',
    'read_json',
    'read_csv',
    'read_h5',
]

def write_dict(
    file_name: str, 
    d: dict, 
    index: Optional[Sequence] = None, 
    attrs: Optional[dict] = None
) -> None:
    """
    write dict to (csv, json, h5)

    Args:
        file_name: the file name to write dict
        d: data dict
        index: use when file_name[-3:] == 'csv'
        attrs: use when file_name[-2:] == 'h5'
    """
    mkdir(os.path.dirname(file_name))

    if file_name[-3:] == 'csv':
        l = len(list(d.values())[0])
        for lst in d.values():
            assert len(lst) == l

        if index is None:
            index = range(1, l + 1)
        pd.DataFrame(d, index=index).to_csv(file_name, float_format='%.3f')

    elif file_name[-4:] == 'json':
        class MyEncoder(json.JSONEncoder):
            def default(self, obj):
                return str(obj)

        with open(file_name, 'w') as f:
            f.write(json.dumps(d, indent=2, separators=(', ', ': '), cls=MyEncoder))

    elif file_name[-2:] == 'h5':
        with h5py.File(file_name, 'w') as f:
            for key, val in d.items():
                f.create_dataset(key, data=val)
                if attrs is not None and key in attrs:
                    f[key].attrs.update(attrs[key])


def read_json(
    file_name: str, 
    type = dict
) -> Optional[dict]:
    """
    read json file

    Args:
        file_name: the file name to read
        type: return data type, must in (dict,)
    """
    assert file_name[-4:] == 'json'

    if type == dict:
        if not os.path.exists(file_name):
            return {}
        with open(file_name, 'r') as f:
            return json.load(f)

def read_csv(
    file_name: str, 
    keys: Optional[str | Sequence[str]] = None, 
    type: Any = pd.DataFrame, 
    dtype: Optional[dict] = None
) -> Any:
    """
    read csv file

    Args:
        file_name: the file name to read
        keys: csv header, if None will return the whole csv
        type: return data type, must in (pd.DataFrame, dict, list)
        dtype: convert csv value type, such as {'file_name': str} will make csv['file_name'].dtype == str
    """
    assert file_name[-3:] == 'csv'

    df = pd.read_csv(file_name, index_col=0, dtype=dtype)
    if keys is not None:
        df = df[keys]
        if isinstance(df, pd.Series):
            df = df.to_frame()
    if type == pd.DataFrame:
        return df
    elif type == dict:
        return {key: df[key].tolist() for key in df}
    elif type == list:
        if df.shape[1] == 1:
            return df[keys].tolist()
        else:
            return [df[key].tolist() for key in df]

def read_h5(
    file_name: str, 
    keys: str | Sequence[str], 
    read_attrs: bool = False
) -> Any:
    """
    read h5
    """
    assert file_name[-2:] == 'h5'

    with h5py.File(file_name, 'r') as f:
        if isinstance(keys, str):
            if read_attrs:
                return f[keys][:], dict(f[keys].attrs)
            else:
                return f[keys][:]
        else:
            vals = [f[key][:] for key in keys]
            if read_attrs:
                attrs = [dict(f[key].attrs) for key in keys]
                return vals, attrs
            else:
                return vals
