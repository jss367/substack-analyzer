"""Test-only synthetic data helpers used by scenarios and unit tests."""

import tempfile

import numpy as np
import pandas as pd


def synthesize_series_with_exog(
    idx: pd.DatetimeIndex,
    K: float,
    r: float,
    exog: pd.Series | None,
    g_exog: float = 0.0,
    step: pd.Series | None = None,
    g_step: float = 0.0,
) -> pd.Series:
    """
    Build a simple logistic-like series with an optional additive exogenous effect on
    month-to-month changes (deltas). Deterministic (no noise) for test stability.
    """
    s_vals: list[float] = [1000.0]
    exog_vals = exog.reindex(idx).astype(float).fillna(0.0).to_numpy() if exog is not None else np.zeros(len(idx))
    step_vals = step.reindex(idx).astype(float).fillna(0.0).to_numpy() if step is not None else np.zeros(len(idx))
    for t in range(1, len(idx)):
        x = s_vals[-1] * (1.0 - s_vals[-1] / K)
        delta = r * x + g_exog * float(exog_vals[t - 1]) + g_step * float(step_vals[t - 1])
        s_vals.append(max(s_vals[-1] + delta, 0.0))
    return pd.Series(np.asarray(s_vals, dtype=float), index=idx, name="Total").round().astype(int)


def ad_spend_csv_with_spikes(idx: pd.DatetimeIndex, spikes: dict) -> str:
    """Write a temp CSV path with zero spend except at specified spike months."""

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
