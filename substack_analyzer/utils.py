"""Shared utility helpers for series handling."""

from collections.abc import Iterable
from typing import Any

import pandas as pd


def ensure_month_end_index(series: pd.Series) -> pd.Series:
    """Return a copy of ``series`` indexed on month-end timestamps.

    The Streamlit app and headless runner both normalise monthly aggregates to
    use month-end ``DatetimeIndex`` values. This helper centralises that logic so
    callers don't need to duplicate the conversion.
    """

    s = series.dropna().copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")
    s.index = s.index.to_period("M").to_timestamp("M")
    s = s.sort_index()
    return s


def coerce_list(values: Any) -> list[Any]:
    """Return ``values`` as a plain ``list``.

    Streamlit session state can hold a variety of container types (including
    ``numpy`` arrays and pandas objects). ``list(x)`` works for many of these,
    but it raises when ``x`` does not define truthiness (e.g. a multi-element
    ``numpy`` array). This helper standardises the conversion so callers don't
    need to guard every access with ``try/except`` blocks.
    """

    if values is None:
        return []
    if isinstance(values, list):
        return list(values)
    if isinstance(values, (tuple, set)):
        return list(values)
    if isinstance(values, (str, bytes)):
        return [values]
    if isinstance(values, Iterable):
        return list(values)
    return [values]
