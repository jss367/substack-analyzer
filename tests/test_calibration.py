import numpy as np
import pandas as pd
import streamlit as st

from substack_analyzer.analysis import build_events_features
from substack_analyzer.calibration import fit_piecewise_logistic, fitted_series_from_params, forecast_piecewise_logistic
from substack_analyzer.changepoints import breakpoints_for_segments, detect_and_classify
from substack_analyzer.detection import detect_change_points
from substack_analyzer.scenarios import cy_series_values, gm_series_values
from substack_analyzer.utils_for_tests import ad_spend_csv_with_spikes, synthesize_series_with_exog


def test_fit_piecewise_logistic_minimal():
    idx = pd.period_range("2023-01", periods=8, freq="M").to_timestamp("M")
    # Simple increasing series
    s = pd.Series(np.linspace(100, 200, num=8), index=idx)
    fit = fit_piecewise_logistic(s, breakpoints=[4])
    assert fit.carrying_capacity > 0
    assert len(fit.segment_growth_rates) == 2
    assert len(fit.fitted_series) == len(s)


def test_forecast_piecewise_logistic_shapes():
    out = forecast_piecewise_logistic(
        last_value=150.0,
        months_ahead=6,
        carrying_capacity=1000.0,
        segment_growth_rate=0.05,
    )
    assert out.shape == (6,)
    assert (out >= 0).all()


def test_fit_piecewise_logistic_requires_minimum_length():
    idx = pd.period_range("2024-01", periods=3, freq="M").to_timestamp("M")
    s = pd.Series([100, 101, 102], index=idx)
    try:
        fit_piecewise_logistic(s, breakpoints=[])
        assert False, "Expected ValueError for series shorter than 4 months"
    except ValueError:
        pass


def test_fit_piecewise_logistic_single_segment_when_no_breakpoints():
    idx = pd.period_range("2023-01", periods=10, freq="M").to_timestamp("M")
    s = pd.Series(np.linspace(50, 150, num=10), index=idx)
    fit = fit_piecewise_logistic(s, breakpoints=[])
    assert len(fit.segment_growth_rates) == 1
    assert fit.carrying_capacity > s.max()


def test_fit_piecewise_logistic_two_segment_rates_ordered():
    # Build a series with faster growth early, slower later
    vals = []
    v = 100
    for _ in range(5):
        vals.append(v)
        v += 20
    for _ in range(5):
        vals.append(v)
        v += 5
    idx = pd.period_range("2023-01", periods=len(vals), freq="M").to_timestamp("M")
    s = pd.Series(vals, index=idx)

    fit = fit_piecewise_logistic(s, breakpoints=[5])
    assert len(fit.segment_growth_rates) == 2
    assert fit.segment_growth_rates[0] > fit.segment_growth_rates[1]


def test_fit_piecewise_logistic_events_reduce_sse():
    # Series with a one-time spike in delta that events can explain
    idx = pd.period_range("2024-01", periods=7, freq="M").to_timestamp("M")
    s = pd.Series([100, 100, 120, 120, 120, 120, 120], index=idx)

    # No events
    fit_no_events = fit_piecewise_logistic(s, breakpoints=[])

    # Transient event at the month of the spike (case-insensitive accepted)
    events_df = pd.DataFrame({"date": [idx[2]], "type": ["promo"], "persistence": ["Transient"]})
    fit_with_events = fit_piecewise_logistic(s, breakpoints=[], events_df=events_df)

    assert fit_with_events.sse < fit_no_events.sse


def test_fit_piecewise_logistic_exogenous_included_and_handles_nans():
    # Construct deltas driven by an exogenous signal
    idx = pd.period_range("2024-01", periods=8, freq="M").to_timestamp("M")
    exog_deltas = [0, 2, 0, 2, 0, 2, 0]
    s_vals = [100]
    for d in exog_deltas:
        s_vals.append(s_vals[-1] + d)
    s = pd.Series(s_vals, index=idx)

    # exog aligned to y index (length len(s)-1), include NaNs which should be treated as zeros
    exog_series = pd.Series([0.0, 1.0, np.nan, 1.0, 0.0, 1.0, 0.0], index=idx[1:])

    fit_without_exog = fit_piecewise_logistic(s, breakpoints=[])
    fit_with_exog = fit_piecewise_logistic(s, breakpoints=[], extra_exog=exog_series)

    assert fit_with_exog.gamma_exog is not None
    assert fit_with_exog.sse < fit_without_exog.sse


def test_fit_piecewise_logistic_three_breaks_mixed_persistence_events():
    """
    Build a synthetic series with three breakpoints and mixed persistence events.
    """
    idx = pd.period_range("2022-01", periods=42, freq="M").to_timestamp("M")

    # Use the simulator to generate a ground-truth series under the same dynamic
    base_series = pd.Series([30.0] * len(idx), index=idx)
    breakpoints = [24, 32, 36]  # three breaks → four segments
    segment_growth_rates = [0.010, 0.017, 0.04, 0.02]

    events_df = pd.DataFrame(
        {
            "date": [idx[20], idx[26], idx[34]],
            "type": ["campaign A", "promo", "campaign B"],
            "persistence": ["persistent", "transient", "persistent"],
        }
    )

    input_series = fitted_series_from_params(
        total_series=base_series,
        breakpoints=breakpoints,
        carrying_capacity=150.0,
        segment_growth_rates=segment_growth_rates,
        events_df=events_df,
        gamma_pulse=3.0,
        gamma_step=0.6,
    )

    # Fit with correct mixed persistence
    fit_mixed = fit_piecewise_logistic(input_series, breakpoints=breakpoints, events_df=events_df)

    # Fit with a mis-specified events table (all transient)
    events_all_transient = events_df.copy()
    events_all_transient["persistence"] = "transient"
    fit_all_transient = fit_piecewise_logistic(input_series, breakpoints=breakpoints, events_df=events_all_transient)

    # Expectations: correct persistence should explain deltas better (lower SSE),
    # both gamma coefficients should be utilized, and four segment rates returned.
    assert len(fit_mixed.segment_growth_rates) == 4
    assert abs(fit_mixed.gamma_step) > 0 or abs(fit_mixed.gamma_pulse) > 0
    assert fit_mixed.sse <= fit_all_transient.sse


def test_fit_piecewise_logistic_on_gm_series():

    input_series = gm_series_values()

    # Detect and classify, then use only segment-worthy breakpoints
    classified = detect_and_classify(input_series, max_changes=4, window=6)
    assert 3 <= len(classified) <= 4
    bkps = breakpoints_for_segments(classified)
    assert 3 <= len(bkps) <= 4
    # Optionally could pass events from classification; not required for this test
    fit = fit_piecewise_logistic(input_series, breakpoints=bkps)

    # Basic shape and plausibility checks
    assert len(fit.fitted_series) == len(input_series)
    assert fit.carrying_capacity > input_series.max()
    assert len(fit.segment_growth_rates) == (len(bkps) + 1 if bkps else 1)
    assert fit.sse <= 1200.0


def test_fit_piecewise_logistic_with_cy_series():
    """
    Slow growth then a faster growth rate
    """
    # Build monthly index
    idx = pd.period_range("2023-09", periods=26, freq="M").to_timestamp("M")

    vals: list[float] = []
    v = 0.0
    jump_month = pd.Timestamp("2025-01-31")
    jump_index = list(idx).index(jump_month)
    for i, ts in enumerate(idx):
        if ts < jump_month:
            v += 80.0 + 1.2 * i
        else:
            v += 80.0 + 1.2 * i + 12 * (i - jump_index)
        vals.append(float(round(v)))

    input_series = pd.Series(vals, index=idx)

    bkps = [16]
    fit = fit_piecewise_logistic(input_series, breakpoints=bkps)

    assert len(fit.fitted_series) == len(input_series)
    assert fit.carrying_capacity > input_series.max()
    assert len(fit.segment_growth_rates) == (len(bkps) + 1 if bkps else 1)
    if len(fit.segment_growth_rates) >= 2:
        # Later growth should be faster than the early growth in this scenario
        assert fit.segment_growth_rates[-1] > fit.segment_growth_rates[0]
    assert fit.sse <= 20


def test_cy_series_ad_effect():
    """
    cy_series_values is driven by a structural growth-rate change, not ads.
    Adding ad exogenous features should have minimal effect on fit quality.
    """
    series = cy_series_values()
    bkps = detect_change_points(series, max_changes=4, min_seg_len=3, return_mode="indices")
    idx = series.index
    plot_df = pd.DataFrame(index=idx)

    # Add some synthetic ad spend spikes unrelated to the structural jump
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, lam=0.5, theta=500.0, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float).fillna(0.0)

    # Fit with and without exogenous regressor
    fit_no_exog = fit_piecewise_logistic(series, breakpoints=bkps)
    fit_with_exog = fit_piecewise_logistic(series, breakpoints=bkps, extra_exog=exog)

    # Ads should not meaningfully explain changes in this scenario
    assert -1 <= fit_with_exog.gamma_exog <= 1  # currently 0.138
    sse_improvement = float(fit_no_exog.sse - fit_with_exog.sse)
    assert sse_improvement / float(fit_no_exog.sse) <= 0.10


def test_phase1_ads_spiky_spend_phase1_json():
    # Monthly timeline
    idx = pd.period_range("2022-01", periods=36, freq="M").to_timestamp("M")
    plot_df = pd.DataFrame(index=idx)

    # Ad spend file with two big spikes
    lam = 0.5
    theta = 500.0
    spikes = {
        idx[6]: 3000.0,  # mid-year push
        idx[18]: 2000.0,  # another big campaign
    }
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    covariates_df, features_df = build_events_features(plot_df, lam=lam, theta=theta, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)

    # Build Total series that uses exogenous effect
    total = synthesize_series_with_exog(idx, K=20000.0, r=0.15, exog=exog, g_exog=100.0)

    # Fit Phase 1 model
    st.session_state.clear()
    st.session_state["import_total"] = total
    st.session_state["import_paid"] = None
    st.session_state["events_df"] = pd.DataFrame()
    st.session_state["covariates_df"] = covariates_df
    st.session_state["adstock_lambda"] = lam
    st.session_state["ad_log_theta"] = theta
    st.session_state["detected_breakpoints"] = []
    st.session_state["detected_change_dates"] = []

    fit = fit_piecewise_logistic(total_series=total, breakpoints=[], events_df=None, extra_exog=exog)

    assert 80 <= fit.gamma_exog <= 120, f"gamma_exog {fit.gamma_exog} is not in range"
    assert fit.r2_on_deltas > 0.95, f"r2_on_deltas {fit.r2_on_deltas} is too low"
    assert fit.sse < 200000, f"sse {fit.sse} is too high"
