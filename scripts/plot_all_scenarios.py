#!/usr/bin/env python3
"""
Render all calibration scenarios into a single image grid.

Output: outputs/all_test_calibration_fits.png
"""


import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from substack_analyzer.analysis import build_events_features
from substack_analyzer.calibration import fit_piecewise_logistic, fitted_series_from_params
from substack_analyzer.changepoints import breakpoints_for_segments, detect_and_classify
from substack_analyzer.plot_utils import plot_fit_vs_actual
from substack_analyzer.scenarios import cy_series_values, gm_series_values, realistic_growth_profiles_cases
from substack_analyzer.utils import ad_spend_csv_with_spikes, synthesize_series_with_exog


def _build_realistic_case_subplot(ax: plt.Axes, case: dict) -> None:
    idx = pd.period_range("2020-01", periods=case["months"], freq="M").to_timestamp("M")
    base_series = pd.Series([case["start"]] * len(idx), index=idx)
    events_df = case["events"](idx)
    total_series = fitted_series_from_params(
        total_series=base_series,
        breakpoints=case["breakpoints"],
        carrying_capacity=case["carrying_capacity"],
        segment_growth_rates=case["segment_rates"],
        events_df=events_df,
        gamma_pulse=case["gamma_pulse"],
        gamma_step=case["gamma_step"],
    )
    fit = fit_piecewise_logistic(
        total_series,
        breakpoints=case["breakpoints"],
        events_df=events_df,
        k_grid=[case["carrying_capacity"]],
    )
    title = f"{case['description']}\nK={fit.carrying_capacity:.0f}, SSE={fit.sse:.1f}, R2Δ={fit.r2_on_deltas:.3f}"
    plot_fit_vs_actual(total_series, fit, title=title, show_breakpoints=True, ax=ax, show=False)


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
    title = f"cy_series (bkps {bkps})\nK={fit.carrying_capacity:.0f}, SSE={fit.sse:.1f}, R2Δ={fit.r2_on_deltas:.3f}"
    plot_fit_vs_actual(series, fit, title=title, show_breakpoints=True, ax=ax, show=False)


def _build_phase1_ads_spiky_subplot(ax: plt.Axes) -> None:
    idx = pd.period_range("2022-01", periods=36, freq="M").to_timestamp("M")
    plot_df = pd.DataFrame(index=idx)
    lam = 0.5
    theta = 500.0
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, lam=lam, theta=theta, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)
    total = synthesize_series_with_exog(idx, K=20000.0, r=0.15, exog=exog, g_exog=100.0)
    fit = fit_piecewise_logistic(total_series=total, breakpoints=[], events_df=None, extra_exog=exog)
    gamma_exog_str = f"{fit.gamma_exog:.1f}" if fit.gamma_exog is not None else "nan"
    title = f"ads_spiky_spend\nγ_exog={gamma_exog_str}, R2Δ={fit.r2_on_deltas:.3f}"
    plot_fit_vs_actual(total, fit, title=title, show_breakpoints=False, ax=ax, show=False)


def main() -> None:
    # Assemble subplot specifications
    builders: list[tuple[str, callable]] = []
    for case in realistic_growth_profiles_cases():
        builders.append((case["description"], lambda ax, case=case: _build_realistic_case_subplot(ax, case)))
    builders.append(("gm_series", _build_gm_series_subplot))
    builders.append(("cy_series", _build_cy_series_subplot))
    builders.append(("ads_spiky_spend", _build_phase1_ads_spiky_subplot))

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
