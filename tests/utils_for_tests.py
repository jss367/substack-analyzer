import tempfile

import numpy as np
import pandas as pd


def synthesize_series_with_exog(
    idx: pd.DatetimeIndex,
    K: float,
    r: float,
    exog: pd.Series | None,
    g_exog: float = 0.0,
) -> pd.Series:
    """
    Build a simple logistic-like series with optional additive exogenous effect on deltas.
    Deterministic (no noise) for test stability.
    """
    s_vals: list[float] = [1000.0]
    exog_vals = exog.reindex(idx).astype(float).fillna(0.0).to_numpy() if exog is not None else np.zeros(len(idx))
    for t in range(1, len(idx)):
        x = s_vals[-1] * (1.0 - s_vals[-1] / float(K))
        delta = r * x + g_exog * float(exog_vals[t - 1])
        s_vals.append(max(s_vals[-1] + delta, 0.0))
    return pd.Series(np.asarray(s_vals, dtype=float), index=idx, name="Total").round().astype(int)


def ad_spend_csv_for_index(idx: pd.DatetimeIndex, monthly_spend: float) -> str:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("date,spend\n")
        for d in idx:
            f.write(f"{d.date()},{monthly_spend}\n")
        return f.name
