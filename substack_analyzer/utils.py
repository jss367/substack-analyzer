"""Shared utility helpers for series handling.

Some of these are mainly used in tests."""

import tempfile

import numpy as np
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


def synthesize_series_with_exog(
    idx: pd.DatetimeIndex,
    K: float,
    r: float,
    exog: pd.Series | None,
    g_exog: float = 0.0,
) -> pd.Series:
    """
    Build a simple logistic-like series with an optional additive exogenous effect on
    month-to-month changes (deltas). Deterministic (no noise) for test stability.

    Dynamics
    --------
    For t >= 1, the increment is computed as:

        delta_t = r * s_{t-1} * (1 - s_{t-1} / K) + g_exog * exog_{t-1}

    and the level updates as:

        s_t = max(s_{t-1} + delta_t, 0)

    Notes on exogenous input
    ------------------------
    - exog is reindexed to `idx`, coerced to float, and NaNs filled with 0.0.
    - The exogenous term at time t uses the previous month's value exog_{t-1} to
      keep the update causal (no peeking ahead).
    - If `exog` is None (or effectively all zeros) or `g_exog == 0.0`, the model
      reduces to a pure logistic update with carrying capacity K and intrinsic
      growth parameter r.

    Choosing g_exog
    ---------------
    - When `exog` is a small-scale feature like log(1 + adstock/theta) (typical
      range ~0–3), values of `g_exog` in the 50–150 range produce a noticeable but
      stable influence in synthetic tests. Use larger values cautiously.

    Parameters
    ----------
    idx : pd.DatetimeIndex
        Monthly index on which the series is generated.
    K : float
        Carrying capacity in the logistic term.
    r : float
        Intrinsic growth coefficient in the logistic term.
    exog : pd.Series | None
        Optional exogenous driver aligned to `idx`. Its magnitude is scaled by
        `g_exog` and added to the monthly delta.
    g_exog : float, default 0.0
        Linear gain applied to the exogenous driver. Set to 0.0 to ignore exog.

    Returns
    -------
    pd.Series
        Generated series named "Total", integer-rounded for readability.
    """
    s_vals: list[float] = [1000.0]
    exog_vals = exog.reindex(idx).astype(float).fillna(0.0).to_numpy() if exog is not None else np.zeros(len(idx))
    for t in range(1, len(idx)):
        x = s_vals[-1] * (1.0 - s_vals[-1] / float(K))
        delta = r * x + g_exog * float(exog_vals[t - 1])
        s_vals.append(max(s_vals[-1] + delta, 0.0))
    return pd.Series(np.asarray(s_vals, dtype=float), index=idx, name="Total").round().astype(int)


def ad_spend_csv_with_spikes(idx: pd.DatetimeIndex, spikes: dict) -> str:
    """Write a temp CSV path with zero spend except at specified spike months.

    spikes keys can be pd.Timestamp, datetime.date, or ISO date strings matching idx dates.
    """

    # Normalize spike keys to date() for direct comparison
    def _normalize_key(k):
        try:
            if isinstance(k, str):
                return pd.to_datetime(k).date()
            if hasattr(k, "date"):
                return k.date()
            return pd.to_datetime(k).date()
        except Exception:
            return None

    norm_spikes = {}
    for k, v in spikes.items():
        nk = _normalize_key(k)
        if nk is not None:
            norm_spikes[nk] = float(v)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("date,spend\n")
        for d in idx:
            amt = norm_spikes.get(d.date(), 0.0)
            f.write(f"{d.date()},{amt}\n")
        return f.name
