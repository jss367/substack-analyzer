import pandas as pd

from substack_analyzer.analysis import build_events_features
from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.detection import detect_change_points
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


def test_phase1_ads_really_valuable_phase1_json():
    # Use the shared scenario to build the synthetic total series
    total = scenario_ads_really_valuable()
    idx = total.index
    plot_df = pd.DataFrame(index=idx)

    # Rebuild exogenous features aligned to the scenario's index
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float).fillna(0.0)

    bkps = detect_change_points(total, max_changes=4, min_seg_len=3, return_mode="indices")

    fit = fit_piecewise_logistic(total_series=total, breakpoints=bkps, events_df=None, extra_exog=exog)

    carrying_capacity = fit.carrying_capacity
    gamma_pulse = fit.gamma_pulse
    gamma_step = fit.gamma_step
    gamma_exog = fit.gamma_exog

    assert 5000 < carrying_capacity < 25000, f"carrying_capacity {carrying_capacity} is not in range"
    assert 0.0 <= gamma_step <= 0.50, f"gamma_step {gamma_step} is not in range"
    assert 0.0 <= gamma_pulse <= 1, f"gamma_pulse {gamma_pulse} is not in range"
    assert 4.0 <= gamma_exog <= 6.5, f"gamma_exog {gamma_exog} is not in range"


def test_phase1_ads_have_no_effect_phase1_json():
    # Use shared scenario to build the series that ignores exogenous effect
    total = scenario_ads_no_effect()
    idx = total.index
    plot_df = pd.DataFrame(index=idx)

    # Rebuild exogenous features aligned to the scenario's index
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float).fillna(0.0)

    bkps = detect_change_points(total, max_changes=4, min_seg_len=3, return_mode="indices")

    fit = fit_piecewise_logistic(total_series=total, breakpoints=bkps, events_df=None, extra_exog=exog)

    carrying_capacity = fit.carrying_capacity
    gamma_intercept = fit.gamma_intercept
    gamma_pulse = fit.gamma_pulse
    gamma_step = fit.gamma_step
    gamma_exog = fit.gamma_exog
    sse = fit.sse

    assert 100 <= carrying_capacity <= 30000, f"carrying_capacity {carrying_capacity} is not in range"
    assert -500.0 <= gamma_intercept <= 500.0, f"gamma_intercept {gamma_intercept} is not in range"
    assert 0.0 <= gamma_pulse <= 1e-3, f"gamma_pulse {gamma_pulse} is not in range"
    assert 0.0 <= gamma_step <= 1e-3, f"gamma_step {gamma_step} is not in range"
    assert -1.2 <= gamma_exog <= 1, f"gamma_exog {gamma_exog} is not in range"  # why is this so close to -1?
    assert sse <= 200


def test_phase1_ads_really_valuable_fit_with_exog():
    # Series synthesized with a strong exogenous ad-effect signal
    series = scenario_ads_really_valuable()

    # Rebuild the same exogenous feature used in the scenario
    idx = series.index
    plot_df = pd.DataFrame(index=idx)
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _cov, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float).fillna(0.0)

    # Fit with exogenous regressor; no structural breakpoints in this scenario
    fit = fit_piecewise_logistic(series, breakpoints=[], extra_exog=exog)

    assert len(fit.fitted_series) == len(series)
    assert fit.carrying_capacity > float(series.max())
    assert len(fit.segment_growth_rates) == 1
    # Autoselection should fall back to either contemporaneous or lagged alignment
    assert fit.exog_lag in (0, 1)
    # Deterministic construction; should explain nearly all variance in deltas
    assert fit.r2_on_deltas > 0.99
    assert fit.sse <= 50


def test_phase1_ads_extremely_valuable_fit_with_exog():
    # Series synthesized with negligible organic growth but extremely strong ad-driven effect
    series = scenario_ads_extremely_valuable()

    # Rebuild the same exogenous feature used in the scenario
    idx = series.index
    plot_df = pd.DataFrame(index=idx)
    spikes = {idx[6]: 15000.0, idx[18]: 20000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _cov, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float).fillna(0.0)

    # Fit with exogenous regressor; no structural breakpoints in this scenario
    fit = fit_piecewise_logistic(series, breakpoints=[], extra_exog=exog)

    assert len(fit.segment_growth_rates) == 1
    assert fit.exog_lag == 1
    assert fit.gamma_exog > 1000
    # The exogenous signal should almost perfectly explain monthly changes
    assert fit.r2_on_deltas > 0.8
    assert fit.sse <= 30_000_000


def test_top_tier_sustained_marketing_series_fit():
    series = top_tier_sustained_marketing_series()
    bkps = [12, 24, 36]
    fit = fit_piecewise_logistic(series, breakpoints=bkps)
    assert len(fit.fitted_series) == len(series)
    assert fit.carrying_capacity > float(series.max())
    assert len(fit.segment_growth_rates) == 4
    assert fit.r2_on_deltas > 0.55
    assert fit.sse <= 52_000_000


def test_small_breakout_series_fit():
    series = small_breakout_series()
    bkps = [18, 30]
    fit = fit_piecewise_logistic(series, breakpoints=bkps)
    assert len(fit.fitted_series) == len(series)
    assert fit.carrying_capacity > float(series.max())
    assert len(fit.segment_growth_rates) == 3
    # Later growth should generally be higher than early growth
    assert fit.segment_growth_rates[-1] > fit.segment_growth_rates[0]
    assert fit.r2_on_deltas > 0.8
    assert fit.sse <= 25_000


def test_niche_steady_series_fit():
    series = niche_steady_series()
    bkps: list[int] = []
    fit = fit_piecewise_logistic(series, breakpoints=bkps)
    assert len(fit.fitted_series) == len(series)
    assert fit.carrying_capacity > float(series.max())
    assert len(fit.segment_growth_rates) == 1
    assert fit.segment_growth_rates[0] > -0.1
    assert fit.r2_on_deltas > 0.02
    assert fit.sse <= 4_000


def test_mid_sized_seasonal_conference_series_fit():
    series = mid_sized_seasonal_conference_series()
    bkps = [10, 20]
    fit = fit_piecewise_logistic(series, breakpoints=bkps)
    assert len(fit.fitted_series) == len(series)
    assert fit.carrying_capacity > float(series.max())
    assert len(fit.segment_growth_rates) == 3
    assert fit.r2_on_deltas > 0.25
    assert fit.sse <= 890_000
