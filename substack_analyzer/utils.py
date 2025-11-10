"""Shared utility helpers for series handling."""

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
