#!/usr/bin/env python3
"""
Render monthly adds (first differences) for ad-impact scenarios, with ad features overlaid.

Output: outputs/monthly_adds_ads.png
"""

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from substack_analyzer.analysis import build_events_features
from substack_analyzer.calibration import fit_piecewise_logistic
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
from substack_analyzer.utils_for_tests import ad_spend_csv_with_spikes

matplotlib.use("Agg")


def _monthly_adds(series: pd.Series) -> pd.Series:
    """First difference; set first month to 0 for clean plotting."""
    adds = series.diff()
    if not adds.empty and pd.isna(adds.iloc[0]):
        adds.iloc[0] = 0.0
    return adds.astype(float)


def _build_ad_features(idx: pd.DatetimeIndex, spikes: dict[pd.Timestamp, float]) -> tuple[pd.Series, pd.Series]:
    """Return (ad_spend, ad_effect_log) aligned to idx using given spikes."""
    plot_df = pd.DataFrame(index=idx)
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    covariates_df, features_df = build_events_features(plot_df, ad_file=ad_file)
    ad_spend = covariates_df["ad_spend"].astype(float)
    ad_effect_log = features_df["ad_effect_log"].astype(float)
    return ad_spend, ad_effect_log


def _plot_adds_with_overlay(
    ax: plt.Axes,
    title: str,
    total_series: pd.Series,
    spikes: dict[pd.Timestamp, float],
    overlay: str = "ad_effect_log",
) -> None:
    """Plot monthly adds and overlay ad feature ('ad_effect_log' or 'ad_spend') on secondary axis."""
    idx = total_series.index
    adds = _monthly_adds(total_series)
    ad_spend, ad_effect_log = _build_ad_features(idx, spikes)
    overlay_series = ad_effect_log if overlay == "ad_effect_log" else ad_spend
    overlay_label = "ad_effect_log" if overlay == "ad_effect_log" else "ad_spend ($)"

    # Optional: fit to report gamma_exog in the title for additional context
    fit = fit_piecewise_logistic(total_series=total_series, breakpoints=[], events_df=None, extra_exog=ad_effect_log)
    gamma_exog = f"{fit.gamma_exog:.2f}" if getattr(fit, "gamma_exog", None) is not None else "nan"
    r2_delta = f"{fit.r2_on_deltas:.3f}" if getattr(fit, "r2_on_deltas", None) is not None else "nan"

    ax.plot(idx, adds, label="monthly adds", color="#1f77b4")
    ax.set_ylabel("Monthly adds")
    ax.set_title(f"{title}\nγ_exog={gamma_exog}, R2Δ={r2_delta}")

    ax2 = ax.twinx()
    ax2.plot(idx, overlay_series, label=overlay_label, color="#DB4437", linestyle="--")
    ax2.set_ylabel(overlay_label)

    # Build a combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")


def _build_ads_really_valuable_subplot(ax: plt.Axes) -> None:
    series = scenario_ads_really_valuable()
    idx = series.index
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    _plot_adds_with_overlay(
        ax=ax,
        title="ads really valuable",
        total_series=series,
        spikes=spikes,
        overlay="ad_effect_log",
    )


def _build_ads_no_effect_subplot(ax: plt.Axes) -> None:
    series = scenario_ads_no_effect()
    idx = series.index
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    _plot_adds_with_overlay(
        ax=ax,
        title="ads no effect",
        total_series=series,
        spikes=spikes,
        overlay="ad_effect_log",
    )


def _build_ads_extremely_valuable_subplot(ax: plt.Axes) -> None:
    series = scenario_ads_extremely_valuable()
    idx = series.index
    spikes = {idx[6]: 15000.0, idx[18]: 20000.0}
    _plot_adds_with_overlay(
        ax=ax,
        title="ads extremely valuable",
        total_series=series,
        spikes=spikes,
        overlay="ad_effect_log",
    )


def _build_comparison_subplot(ax: plt.Axes) -> None:
    """Overlay monthly adds for 'really valuable' vs 'no effect' with ad feature on right axis."""
    s_val = scenario_ads_really_valuable()
    s_no = scenario_ads_no_effect()
    idx = s_val.index

    adds_val = _monthly_adds(s_val)
    adds_no = _monthly_adds(s_no)

    # Shared ad schedule for these two scenarios
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    _, ad_effect_log = _build_ad_features(idx, spikes)

    # Fit γ_exog for the 'valuable' path for context
    fit_val = fit_piecewise_logistic(total_series=s_val, breakpoints=[], events_df=None, extra_exog=ad_effect_log)
    gamma_exog = f"{fit_val.gamma_exog:.2f}" if getattr(fit_val, "gamma_exog", None) is not None else "nan"

    ax.plot(idx, adds_val, label="monthly adds (valuable)", color="#2ca02c")
    ax.plot(idx, adds_no, label="monthly adds (no effect)", color="#7f7f7f", linestyle=":")
    ax.set_ylabel("Monthly adds")
    ax.set_title(f"valuable vs no-effect (adds)\nγ_exog={gamma_exog}")

    ax2 = ax.twinx()
    ax2.plot(idx, ad_effect_log, label="ad_effect_log", color="#DB4437", linestyle="--")
    ax2.set_ylabel("ad_effect_log")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")


def _plot_adds_simple(ax: plt.Axes, title: str, total_series: pd.Series) -> None:
    """Plot monthly adds without any overlay."""
    idx = total_series.index
    adds = _monthly_adds(total_series)
    ax.plot(idx, adds, label="monthly adds", color="#1f77b4")
    ax.set_ylabel("Monthly adds")
    ax.set_title(title)
    ax.legend(loc="upper left")


def _build_top_tier_adds_subplot(ax: plt.Axes) -> None:
    series = top_tier_sustained_marketing_series()
    _plot_adds_simple(ax, "top_tier_sustained_marketing (adds)", series)


def _build_small_breakout_adds_subplot(ax: plt.Axes) -> None:
    series = small_breakout_series()
    _plot_adds_simple(ax, "small_breakout (adds)", series)


def _build_niche_steady_adds_subplot(ax: plt.Axes) -> None:
    series = niche_steady_series()
    _plot_adds_simple(ax, "niche_steady (adds)", series)


def _build_mid_sized_adds_subplot(ax: plt.Axes) -> None:
    series = mid_sized_seasonal_conference_series()
    _plot_adds_simple(ax, "mid_sized_seasonal_conference (adds)", series)


def _build_gm_adds_subplot(ax: plt.Axes) -> None:
    series = gm_series_values()
    _plot_adds_simple(ax, "gm_series (adds)", series)


def _build_cy_adds_subplot(ax: plt.Axes) -> None:
    series = cy_series_values()
    _plot_adds_simple(ax, "cy_series (adds)", series)


def _build_one_time_spike_adds_subplot(ax: plt.Axes) -> None:
    series, _events = one_time_spike_series_and_events()
    _plot_adds_simple(ax, "one_time_spike (adds)", series)


def main() -> None:
    builders: list[tuple[str, callable]] = [
        # Core scenarios (adds only)
        ("top_tier_adds", _build_top_tier_adds_subplot),
        ("small_breakout_adds", _build_small_breakout_adds_subplot),
        ("niche_steady_adds", _build_niche_steady_adds_subplot),
        ("mid_sized_adds", _build_mid_sized_adds_subplot),
        ("gm_adds", _build_gm_adds_subplot),
        ("cy_adds", _build_cy_adds_subplot),
        ("one_time_spike_adds", _build_one_time_spike_adds_subplot),
        # Ad scenarios (adds + ad overlay)
        ("ads_really_valuable", _build_ads_really_valuable_subplot),
        ("ads_no_effect", _build_ads_no_effect_subplot),
        ("ads_extremely_valuable", _build_ads_extremely_valuable_subplot),
        ("comparison", _build_comparison_subplot),
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
    out_path = out_dir / "monthly_adds_ads.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
