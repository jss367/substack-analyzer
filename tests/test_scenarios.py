import json

import pandas as pd
import streamlit as st

from substack_analyzer.analysis import build_events_features
from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.persistence import export_phase_one_json
from tests.utils_for_tests import ad_spend_csv_for_index, synthesize_series_with_exog


def test_phase1_ads_really_valuable_phase1_json():
    # Monthly timeline
    idx = pd.period_range("2022-01", periods=36, freq="M").to_timestamp("M")
    plot_df = pd.DataFrame(index=idx)

    # Ad spend file (constant spend) -> features with ad_effect_log
    lam = 0.5
    theta = 500.0
    ad_file = ad_spend_csv_for_index(idx, monthly_spend=5000.0)
    covariates_df, features_df = build_events_features(plot_df, lam=lam, theta=theta, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)

    # Build Total series that actually uses exogenous effect (positive influence)
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
    st.session_state["pwlog_fit"] = fit

    # Export Phase 1 JSON
    data = export_phase_one_json()
    obj = json.loads(data)
    fp = obj.get("fit_params")
    assert isinstance(fp, dict)

    # Rounding and typing
    assert 10000 < fp["carrying_capacity"] < 30000
    for x in fp["segment_growth_rates"]:
        assert 0.10 < x < 0.50
    for k in ["gamma_pulse", "gamma_step", "gamma_intercept"]:
        assert abs(fp[k] - round(fp[k], 6)) == 0
    # Reasonable ranges given no pulse/step events in this scenario
    assert abs(fp["gamma_pulse"]) <= 1e-3
    assert abs(fp["gamma_step"]) <= 1e-3
    # Intercept should be modest; synthetic series has no explicit intercept
    assert -10.0 <= fp["gamma_intercept"] <= 10.0

    # Valuable ads: gamma_exog should be positive and of reasonable magnitude
    gx = fp.get("gamma_exog")
    assert gx is not None and isinstance(gx, float)
    # Export rounds to 6 decimals
    assert abs(gx - round(gx, 6)) == 0
    # In this synthetic setup, true g_exog=100 and ad_effect_log ~ O(1-5)
    # Keep a conservative band around 100
    assert 50.0 <= gx <= 200.0


def test_phase1_ads_have_no_effect_phase1_json():
    # Monthly timeline
    idx = pd.period_range("2022-01", periods=36, freq="M").to_timestamp("M")
    plot_df = pd.DataFrame(index=idx)

    # Ad spend file present, but series generation ignores exog (g_exog=0)
    lam = 0.5
    theta = 500.0
    ad_file = ad_spend_csv_for_index(idx, monthly_spend=5000.0)
    covariates_df, features_df = build_events_features(plot_df, lam=lam, theta=theta, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)

    # Build Total series without exogenous effect
    total = synthesize_series_with_exog(idx, K=20000.0, r=0.15, exog=exog, g_exog=0.0)

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
    st.session_state["pwlog_fit"] = fit

    # Export Phase 1 JSON
    data = export_phase_one_json()
    obj = json.loads(data)
    fp = obj.get("fit_params")
    assert isinstance(fp, dict)

    # Rounding and typing
    assert isinstance(fp["carrying_capacity"], int)
    for x in fp["segment_growth_rates"]:
        assert abs(x - round(x, 6)) == 0
    for k in ["gamma_pulse", "gamma_step", "gamma_intercept"]:
        assert abs(fp[k] - round(fp[k], 6)) == 0
    # Reasonable ranges given no pulse/step events in this scenario
    assert abs(fp["gamma_pulse"]) <= 1e-3
    assert abs(fp["gamma_step"]) <= 1e-3
    # Intercept should be modest; synthetic series has no explicit intercept
    assert -10.0 <= fp["gamma_intercept"] <= 10.0

    # No effect: gamma_exog should round to 0.0
    gx = fp.get("gamma_exog")
    assert gx is None or isinstance(gx, float)
    # Export rounds to 6 decimals; enforce rounding consistency and near-zero magnitude
    assert abs(gx - round(gx, 6)) == 0
    assert abs(gx) <= 1e-6
