import pandas as pd
from collections.abc import Mapping
from datetime import tzinfo
from typing import Optional


def to_frame(obj, name: str, common_index=None, fill_value=False):
    """Convert Series/DataFrame/dict-like into a DataFrame.

    Parameters
    ----------
    obj : DataFrame | Series | Mapping
        Object to convert.
    name : str
        Name used for column or error messages.
    common_index : Index, optional
        Index to reindex the result to.
    fill_value : Any, default False
        Fill value used when reindexing.
    """
    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, Mapping):
        df = pd.DataFrame({k: (v if isinstance(v, pd.Series) else v.squeeze())
                           for k, v in obj.items()})
    elif isinstance(obj, pd.Series):
        df = obj.to_frame(name=name)
    else:
        raise TypeError(f"{name} must be DataFrame/Series/dict-like; got {type(obj).__name__}")
    if common_index is not None:
        df = df.reindex(common_index, fill_value=fill_value)
    return df


def assert_monotonic_datetime_index(df: pd.DataFrame, name: str = "DataFrame") -> Optional[tzinfo]:
    """Ensure ``df`` has a monotonic ``DatetimeIndex``.

    Parameters
    ----------
    df : DataFrame
        Frame to validate.
    name : str, default "DataFrame"
        Human readable name used in error messages.

    Returns
    -------
    tz : tzinfo or None
        The timezone of the index for cross-checking across multiple frames.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonic increasing")
    return df.index.tz
