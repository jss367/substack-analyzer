#!/usr/bin/env python3
"""
Render all calibration scenarios into a single image grid.

Output: outputs/all_test_calibration_fits.png
"""


import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from substack_analyzer.analysis import build_events_features
from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.changepoints import breakpoints_for_segments, detect_and_classify
from substack_analyzer.plot_utils import plot_fit_vs_actual
from substack_analyzer.scenarios import (
    cy_series_values,
    gm_series_values,
    mid_sized_seasonal_conference_series,
    niche_steady_series,
    one_time_spike_series_and_events,
    scenario_ads_extremely_valuable,
    scenario_ads_no_effect,
    scenario_ads_really_valuable,
    small_breakout_series,
    top_tier_sustained_marketing_series,
)
from substack_analyzer.utils_for_tests import ad_spend_csv_with_spikes, synthesize_series_with_exog

matplotlib.use("Agg")


def _build_series_with_bkps(ax: plt.Axes, title: str, series: pd.Series, bkps: list[int] | None) -> None:
    fit = fit_piecewise_logistic(series, breakpoints=bkps or [])
    subtitle = f"K={fit.carrying_capacity:.0f}, SSE={fit.sse:.1f}, R2Δ={fit.r2_on_deltas:.3f}"
    plot_fit_vs_actual(series, fit, title=f"{title}\n{subtitle}", show_breakpoints=True, ax=ax, show=False)


def _build_top_tier_subplot(ax: plt.Axes) -> None:
    series = top_tier_sustained_marketing_series()
    _build_series_with_bkps(ax, "top_tier_sustained_marketing", series, [12, 24, 36])


def _build_small_breakout_subplot(ax: plt.Axes) -> None:
    series = small_breakout_series()
    _build_series_with_bkps(ax, "small_breakout", series, [18, 30])


def _build_niche_steady_subplot(ax: plt.Axes) -> None:
    series = niche_steady_series()
    _build_series_with_bkps(ax, "niche_steady", series, [])


def _build_mid_sized_subplot(ax: plt.Axes) -> None:
    series = mid_sized_seasonal_conference_series()
    _build_series_with_bkps(ax, "mid_sized_seasonal_conference", series, [10, 20])


def _build_gm_series_subplot(ax: plt.Axes) -> None:
    series = gm_series_values()
    classified = detect_and_classify(series, max_changes=4, window=6)
    bkps = breakpoints_for_segments(classified)
    fit = fit_piecewise_logistic(series, breakpoints=bkps)
    title = (
        f"gm_series (auto bkps {bkps})\nK={fit.carrying_capacity:.0f}, SSE={fit.sse:.1f}, R2Δ={fit.r2_on_deltas:.3f}"
    )
    plot_fit_vs_actual(series, fit, title=title, show_breakpoints=True, ax=ax, show=False)


def _build_cy_series_subplot(ax: plt.Axes) -> None:
    series, bkps = cy_series_values()
    fit = fit_piecewise_logistic(series, breakpoints=bkps)
    title = f"cy_series (bkps {bkps})\n" f"K={fit.carrying_capacity:.0f}, SSE={fit.sse:.1f}, R2Δ={fit.r2_on_deltas:.3f}"
    plot_fit_vs_actual(series, fit, title=title, show_breakpoints=True, ax=ax, show=False)


def _build_phase1_ads_spiky_subplot(ax: plt.Axes) -> None:
    idx = pd.period_range("2022-01", periods=36, freq="M").to_timestamp("M")
    plot_df = pd.DataFrame(index=idx)
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)
    total = synthesize_series_with_exog(idx, K=20000.0, r=0.15, exog=exog, g_exog=100.0)
    fit = fit_piecewise_logistic(total_series=total, breakpoints=[], events_df=None, extra_exog=exog)
    gamma_exog = f"{fit.gamma_exog:.1f}" if fit.gamma_exog is not None else "nan"
    title = f"ads_spiky_spend\n" f"γ_exog={gamma_exog}, R2Δ={fit.r2_on_deltas:.3f}"
    plot_fit_vs_actual(total, fit, title=title, show_breakpoints=False, ax=ax, show=False)


def _build_phase1_ads_valuable_subplot(ax: plt.Axes) -> None:
    # Scenario with constant monthly ad spend whose effect is modeled via exogenous features
    series = scenario_ads_really_valuable()
    idx = series.index
    plot_df = pd.DataFrame(index=idx)
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)
    fit = fit_piecewise_logistic(total_series=series, breakpoints=[], events_df=None, extra_exog=exog)
    gamma_exog = f"{fit.gamma_exog:.1f}" if fit.gamma_exog is not None else "nan"
    title = f"ads really valuable\n" f"γ_exog={gamma_exog}, R2Δ={fit.r2_on_deltas:.3f}"
    plot_fit_vs_actual(series, fit, title=title, show_breakpoints=False, ax=ax, show=False)


def _build_phase1_ads_no_effect_subplot(ax: plt.Axes) -> None:
    # Scenario with ad spend present but total ignores exogenous effect
    series = scenario_ads_no_effect()
    idx = series.index
    plot_df = pd.DataFrame(index=idx)
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)
    fit = fit_piecewise_logistic(total_series=series, breakpoints=[], events_df=None, extra_exog=exog)
    gamma_exog = f"{fit.gamma_exog:.1f}" if fit.gamma_exog is not None else "nan"
    title = f"ads no effect\n" f"γ_exog={gamma_exog}, R2Δ={fit.r2_on_deltas:.3f}"
    plot_fit_vs_actual(series, fit, title=title, show_breakpoints=False, ax=ax, show=False)


def _build_phase1_ads_extremely_valuable_subplot(ax: plt.Axes) -> None:
    # Scenario with negligible organic growth but massive ad-driven growth
    series = scenario_ads_extremely_valuable()
    idx = series.index
    plot_df = pd.DataFrame(index=idx)
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)
    fit = fit_piecewise_logistic(total_series=series, breakpoints=[], events_df=None, extra_exog=exog)
    gamma_exog = f"{fit.gamma_exog:.1f}" if fit.gamma_exog is not None else "nan"
    title = f"ads extremely valuable\n" f"γ_exog={gamma_exog}, R2Δ={fit.r2_on_deltas:.3f}"
    plot_fit_vs_actual(series, fit, title=title, show_breakpoints=False, ax=ax, show=False)


def _build_one_time_spike_subplot(ax: plt.Axes) -> None:
    series, events = one_time_spike_series_and_events()
    fit = fit_piecewise_logistic(series, breakpoints=[], events_df=events)
    title = f"one_time_spike (events)\n" f"K={fit.carrying_capacity:.0f}, SSE={fit.sse:.1f}, R2Δ={fit.r2_on_deltas:.3f}"
    plot_fit_vs_actual(series, fit, title=title, show_breakpoints=False, ax=ax, show=False)


def main() -> None:
    # Assemble subplot specifications
    builders: list[tuple[str, callable]] = [
        ("top_tier", _build_top_tier_subplot),
        ("small_breakout", _build_small_breakout_subplot),
        ("niche_steady", _build_niche_steady_subplot),
        ("mid_sized", _build_mid_sized_subplot),
        ("gm_series", _build_gm_series_subplot),
        ("cy_series", _build_cy_series_subplot),
        ("one_time_spike", _build_one_time_spike_subplot),
        ("ads_spiky_spend", _build_phase1_ads_spiky_subplot),
        ("ads_constant_spend", _build_phase1_ads_valuable_subplot),
        ("ads_no_effect", _build_phase1_ads_no_effect_subplot),
        ("ads_extreme_value", _build_phase1_ads_extremely_valuable_subplot),
    ]

    n = len(builders)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 4.8 * rows), squeeze=False)

    for i, (_, builder) in enumerate(builders):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        builder(ax)

    # Hide any unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_test_calibration_fits.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
