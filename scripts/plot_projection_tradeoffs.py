#!/usr/bin/env python3
"""Visualize forward-projection tradeoffs for intercept and exogenous terms."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from substack_analyzer.analysis import build_events_features
from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.scenarios import niche_steady_series, scenario_ads_really_valuable
from substack_analyzer.utils_for_tests import ad_spend_csv_with_spikes

matplotlib.use("Agg")


def _extend_monthly_index(index: pd.DatetimeIndex, months_ahead: int) -> pd.DatetimeIndex:
    """Return a new month-end index extending ``index`` by ``months_ahead`` months."""

    if index.empty:
        raise ValueError("Cannot extend an empty index")
    last = index[-1]
    freq = pd.infer_freq(index)
    if freq is None:
        # The synthetic series in scenarios.py are all month-end; enforce explicitly.
        freq = "M"
    future = pd.date_range(last + pd.offsets.MonthEnd(1), periods=months_ahead, freq=freq)
    return future


def _forecast_with_terms(
    last_value: float,
    months_ahead: int,
    carrying_capacity: float,
    growth_rate: float,
    intercept: float = 0.0,
    gamma_exog: float | None = None,
    exog_future: np.ndarray | None = None,
) -> np.ndarray:
    """Simple forward simulation that mirrors the final-segment dynamics."""

    values: list[float] = []
    current = float(last_value)
    exog_future = np.asarray(exog_future, dtype=float) if exog_future is not None else None
    for step in range(months_ahead):
        base = current * (1.0 - current / carrying_capacity)
        delta = growth_rate * base + intercept
        if gamma_exog is not None and exog_future is not None and step < len(exog_future):
            delta += gamma_exog * float(exog_future[step])
        current = max(current + delta, 0.0)
        values.append(current)
    return np.asarray(values, dtype=float)


def _plot_intercept_tradeoff(ax: plt.Axes, months_ahead: int = 24) -> None:
    series = niche_steady_series()
    fit = fit_piecewise_logistic(series, breakpoints=[])

    k = fit.carrying_capacity
    r_last = fit.segment_growth_rates[-1]
    intercept_last = fit.segment_intercepts[-1] if fit.segment_intercepts else fit.gamma_intercept

    future_idx = _extend_monthly_index(series.index, months_ahead)
    baseline = _forecast_with_terms(series.iat[-1], months_ahead, k, r_last)
    with_intercept = _forecast_with_terms(series.iat[-1], months_ahead, k, r_last, intercept=intercept_last)

    ax.plot(series.index, series.to_numpy(dtype=float), label="Actual history", color="black", linewidth=1.6)
    ax.plot(future_idx, baseline, label="Forecast: logistic only", linestyle="--", color="#1f77b4")
    ax.plot(
        future_idx,
        with_intercept,
        label=f"Forecast: + intercept ({intercept_last:.0f}/mo)",
        linestyle="-.",
        color="#ff7f0e",
    )
    ax.set_title("Effect of keeping fitted intercept")
    ax.set_ylabel("Subscribers")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def _plot_exogenous_tradeoff(ax: plt.Axes, months_ahead: int = 24) -> None:
    series = scenario_ads_really_valuable()
    idx = series.index
    plot_df = pd.DataFrame(index=idx)
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)

    fit = fit_piecewise_logistic(series, breakpoints=[], extra_exog=exog)

    k = fit.carrying_capacity
    r_last = fit.segment_growth_rates[-1]
    intercept_last = fit.segment_intercepts[-1] if fit.segment_intercepts else fit.gamma_intercept
    gamma_exog = fit.gamma_exog or 0.0

    months = months_ahead
    future_idx = _extend_monthly_index(series.index, months)

    baseline = _forecast_with_terms(series.iat[-1], months, k, r_last)
    intercept_only = _forecast_with_terms(series.iat[-1], months, k, r_last, intercept=intercept_last)

    last_exog = float(exog.iloc[-1]) if not exog.empty else 0.0
    avg_exog = float(exog.tail(6).mean()) if exog.size else 0.0
    future_exog_continue = np.full(months, avg_exog if np.isfinite(avg_exog) else last_exog)
    with_ads = _forecast_with_terms(
        series.iat[-1],
        months,
        k,
        r_last,
        intercept=intercept_last,
        gamma_exog=gamma_exog,
        exog_future=future_exog_continue,
    )

    ax.plot(series.index, series.to_numpy(dtype=float), label="Actual history", color="black", linewidth=1.6)
    ax.plot(future_idx, baseline, label="Forecast: logistic only", linestyle="--", color="#1f77b4")
    ax.plot(
        future_idx,
        intercept_only,
        label="Forecast: + intercept",
        linestyle="-.",
        color="#ff7f0e",
    )
    ax.plot(
        future_idx,
        with_ads,
        label="Forecast: + intercept + ads continue",
        linestyle=":",
        color="#2ca02c",
    )
    ax.set_title("Role of exogenous ad term")
    ax.set_ylabel("Subscribers")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax.annotate(
        f"γ_exog = {gamma_exog:.2f}\nAssumed future ad_effect_log ≈ {future_exog_continue[0]:.2f}",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.6),
    )


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _plot_intercept_tradeoff(axes[0])
    _plot_exogenous_tradeoff(axes[1])
    axes[0].set_xlabel("Month")
    axes[1].set_xlabel("Month")
    plt.tight_layout()
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "projection_tradeoffs.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
