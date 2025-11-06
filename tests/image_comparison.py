import matplotlib
import pandas as pd
import streamlit as st
from utils_for_tests import ad_spend_csv_for_index, ad_spend_csv_with_spikes, synthesize_series_with_exog

from substack_analyzer.analysis import build_events_features
from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.plot_utils import plot_fit_vs_actual

matplotlib.use("TkAgg")


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

    plot_fit_vs_actual(total, fit)


if __name__ == "__main__":
    test_phase1_ads_really_valuable_phase1_json()
