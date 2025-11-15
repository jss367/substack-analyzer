#!/usr/bin/env python3
"""Visualize forward-projection tradeoffs for intercept and exogenous terms."""

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from substack_analyzer.analysis import build_events_features
from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.scenarios import (
    mid_sized_seasonal_conference_series,
    niche_steady_series,
    scenario_ads_extremely_valuable,
    scenario_ads_no_effect,
    scenario_ads_really_valuable,
    small_breakout_series,
    top_tier_sustained_marketing_series,
)
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
    return pd.date_range(last + pd.offsets.MonthEnd(1), periods=months_ahead, freq=freq)


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
    current = last_value
    exog_future = np.asarray(exog_future, dtype=float) if exog_future is not None else None
    for step in range(months_ahead):
        base = current * (1.0 - current / carrying_capacity)
        delta = growth_rate * base + intercept
        if gamma_exog is not None and exog_future is not None and step < len(exog_future):
            delta += gamma_exog * float(exog_future[step])
        current = max(current + delta, 0.0)
        values.append(current)
    return np.asarray(values, dtype=float)


def _plot_intercept_tradeoff(
    ax: plt.Axes, series: pd.Series, title: str, breakpoints: list[int] | None = None, months_ahead: int = 24
) -> None:
    """Plot intercept tradeoff for a given series."""
    fit = fit_piecewise_logistic(series, breakpoints=breakpoints or [])

    k = fit.carrying_capacity
    r_last = fit.segment_growth_rates[-1]
    intercept_last = fit.segment_intercepts[-1] if fit.segment_intercepts else fit.gamma_intercept

    future_idx = _extend_monthly_index(series.index, months_ahead)
    with_intercept = _forecast_with_terms(series.iat[-1], months_ahead, k, r_last, intercept=intercept_last)

    ax.plot(series.index, series.to_numpy(dtype=float), label="Actual history", color="black", linewidth=1.6)
    ax.plot(
        future_idx,
        with_intercept,
        label=f"Forecast (+ intercept {intercept_last:.0f}/mo)",
        linestyle="-.",
        color="#1f77b4",
    )
    ax.set_title(title)
    ax.set_ylabel("Subscribers")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_exogenous_tradeoff(
    ax: plt.Axes,
    series: pd.Series,
    title: str,
    spikes: dict | None = None,
    months_ahead: int = 24,
) -> None:
    """Plot exogenous tradeoff for a given series with ad spend."""
    idx = series.index
    plot_df = pd.DataFrame(index=idx)
    if spikes is None:
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

    intercept_only = _forecast_with_terms(series.iat[-1], months, k, r_last, intercept=intercept_last)

    last_exog = float(exog.iloc[-1]) if exog.size > 0 else 0.0
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
    ax.plot(
        future_idx,
        intercept_only,
        label="Forecast (+ intercept)",
        linestyle="-.",
        color="#1f77b4",
    )
    ax.plot(
        future_idx,
        with_ads,
        label="Forecast: + intercept + ads continue",
        linestyle=":",
        color="#2ca02c",
    )
    ax.set_title(title)
    ax.set_ylabel("Subscribers")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.annotate(
        f"γ_exog = {gamma_exog:.2f}\nAssumed future ad_effect_log ≈ {future_exog_continue[0]:.2f}",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.6),
    )


def main() -> None:
    # Define builder functions for each scenario
    def _build_niche_steady(ax: plt.Axes) -> None:
        _plot_intercept_tradeoff(ax, niche_steady_series(), "niche_steady")

    def _build_small_breakout(ax: plt.Axes) -> None:
        _plot_intercept_tradeoff(ax, small_breakout_series(), "small_breakout", breakpoints=[18, 30])

    def _build_top_tier(ax: plt.Axes) -> None:
        _plot_intercept_tradeoff(
            ax, top_tier_sustained_marketing_series(), "top_tier_sustained_marketing", breakpoints=[12, 24, 36]
        )

    def _build_mid_sized(ax: plt.Axes) -> None:
        _plot_intercept_tradeoff(ax, mid_sized_seasonal_conference_series(), "mid_sized_seasonal", breakpoints=[10, 20])

    def _build_ads_really_valuable(ax: plt.Axes) -> None:
        series = scenario_ads_really_valuable()
        idx = series.index
        _plot_exogenous_tradeoff(ax, series, "ads_really_valuable", spikes={idx[6]: 3000.0, idx[18]: 2000.0})

    def _build_ads_extremely_valuable(ax: plt.Axes) -> None:
        series = scenario_ads_extremely_valuable()
        idx = series.index
        _plot_exogenous_tradeoff(ax, series, "ads_extremely_valuable", spikes={idx[6]: 15000.0, idx[18]: 20000.0})

    def _build_ads_no_effect(ax: plt.Axes) -> None:
        series = scenario_ads_no_effect()
        idx = series.index
        _plot_exogenous_tradeoff(ax, series, "ads_no_effect", spikes={idx[6]: 3000.0, idx[18]: 2000.0})

    builders: list[tuple[str, callable]] = [
        ("Intercept: niche_steady", _build_niche_steady),
        ("Intercept: small_breakout", _build_small_breakout),
        ("Intercept: top_tier", _build_top_tier),
        ("Intercept: mid_sized", _build_mid_sized),
        ("Exog: ads_really_valuable", _build_ads_really_valuable),
        ("Exog: ads_extremely_valuable", _build_ads_extremely_valuable),
        ("Exog: ads_no_effect", _build_ads_no_effect),
    ]

    n = len(builders)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16 * cols / 3, 5 * rows), squeeze=False)

    for i, (_, builder) in enumerate(builders):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        builder(ax)
        ax.set_xlabel("Month")

    # Hide any unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "projection_tradeoffs.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
