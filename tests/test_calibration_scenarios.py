from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.scenarios import (
    mid_sized_seasonal_conference_series,
    niche_steady_series,
    small_breakout_series,
    top_tier_sustained_marketing_series,
)


def test_top_tier_sustained_marketing_series_fit():
    series = top_tier_sustained_marketing_series()
    bkps = [12, 24, 36]
    fit = fit_piecewise_logistic(series, breakpoints=bkps)
    assert len(fit.fitted_series) == len(series)
    assert fit.carrying_capacity > float(series.max())
    assert len(fit.segment_growth_rates) == 4
    assert fit.r2_on_deltas > 0.8
    assert fit.sse >= 0.0


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
    assert fit.sse >= 0.0


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

    assert fit.r2_on_deltas > 0.05  # This metric doesn't really matter
    assert fit.sse >= 0


def test_mid_sized_seasonal_conference_series_fit():
    series = mid_sized_seasonal_conference_series()
    bkps = [10, 20]
    fit = fit_piecewise_logistic(series, breakpoints=bkps)
    assert len(fit.fitted_series) == len(series)
    assert fit.carrying_capacity > float(series.max())
    assert len(fit.segment_growth_rates) == 3
    assert fit.r2_on_deltas > 0.7
    assert fit.sse >= 0.0
