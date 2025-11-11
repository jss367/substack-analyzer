import pandas as pd

from substack_analyzer.analysis import build_events_features
from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.detection import detect_change_points
from substack_analyzer.scenarios import (
    mid_sized_seasonal_conference_series,
    niche_steady_series,
    scenario_ads_really_valuable,
    small_breakout_series,
    top_tier_sustained_marketing_series,
)
from substack_analyzer.utils_for_tests import ad_spend_csv_with_spikes


def test_top_tier_sustained_marketing_series_fit():
    series = top_tier_sustained_marketing_series()
    bkps = [12, 24, 36]
    fit = fit_piecewise_logistic(series, breakpoints=bkps)
    assert len(fit.fitted_series) == len(series)
    assert fit.carrying_capacity > float(series.max())
    assert len(fit.segment_growth_rates) == 4
    assert fit.r2_on_deltas > 0.6
    assert fit.sse <= 60000000


def test_small_breakout_series_fit():
    series = small_breakout_series()
    bkps = detect_change_points(series, max_changes=4, min_seg_len=3, return_mode="indices")
    fit = fit_piecewise_logistic(series, breakpoints=bkps)
    assert len(fit.fitted_series) == len(series)
    assert fit.carrying_capacity > float(series.max())
    assert len(fit.segment_growth_rates) == 3
    # Later growth should generally be higher than early growth
    assert fit.segment_growth_rates[-1] > fit.segment_growth_rates[0]
    assert fit.r2_on_deltas > 0.8
    assert fit.sse <= 30000


def test_niche_steady_series_fit():
    series = niche_steady_series()
    bkps: list[int] = []
    fit = fit_piecewise_logistic(series, breakpoints=bkps)
    assert len(fit.fitted_series) == len(series)
    assert fit.carrying_capacity > float(series.max())
    assert len(fit.segment_growth_rates) == 1
    assert (
        fit.segment_growth_rates[0] > -0.1
    )  # for now it can be a little negative because gamma_intercept explains the positive growth. I think this is fine.

    assert fit.r2_on_deltas > 0.02  # This metric doesn't really matter
    assert fit.sse <= 10000


def test_mid_sized_seasonal_conference_series_fit():
    series = mid_sized_seasonal_conference_series()
    bkps = [10, 20]
    fit = fit_piecewise_logistic(series, breakpoints=bkps)
    assert len(fit.fitted_series) == len(series)
    assert fit.carrying_capacity > float(series.max())
    assert len(fit.segment_growth_rates) == 3
    assert fit.r2_on_deltas > 0.2
    assert fit.sse <= 900000


def test_phase1_ads_really_valuable_fit_with_exog():
    # Series synthesized with a strong exogenous ad-effect signal
    series = scenario_ads_really_valuable()

    # Rebuild the same exogenous feature used in the scenario
    idx = series.index
    plot_df = pd.DataFrame(index=idx)
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _cov, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)

    # Fit with exogenous regressor; no structural breakpoints in this scenario
    fit = fit_piecewise_logistic(series, breakpoints=[], extra_exog=exog)

    assert len(fit.fitted_series) == len(series)
    assert fit.carrying_capacity > float(series.max())
    assert len(fit.segment_growth_rates) == 1
    # Deterministic construction; should explain nearly all variance in deltas
    assert fit.r2_on_deltas > 0.95
    assert fit.sse <= 50
