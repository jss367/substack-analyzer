import pandas as pd

from substack_analyzer.analysis import build_events_features
from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.detection import detect_change_points
from substack_analyzer.scenarios import test_phase1_ads_have_no_effect_phase1_json as scenario_ads_no_effect
from substack_analyzer.scenarios import test_phase1_ads_really_valuable_phase1_json as scenario_ads_really_valuable
from substack_analyzer.utils import ad_spend_csv_with_spikes


def test_phase1_ads_really_valuable_phase1_json():
    # Use the shared scenario to build the synthetic total series
    total = scenario_ads_really_valuable()
    idx = total.index
    plot_df = pd.DataFrame(index=idx)

    # Rebuild exogenous features aligned to the scenario's index
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)

    bkps = detect_change_points(total, max_changes=4, min_seg_len=3, return_mode="indices")

    fit = fit_piecewise_logistic(total_series=total, breakpoints=bkps, events_df=None, extra_exog=exog)

    carrying_capacity = fit.carrying_capacity
    gamma_intercept = fit.gamma_intercept
    gamma_pulse = fit.gamma_pulse
    gamma_step = fit.gamma_step
    gamma_exog = fit.gamma_exog

    assert 10000 < carrying_capacity < 30000, f"carrying_capacity {carrying_capacity} is not in range"
    assert 0 < gamma_step < 0.50, f"gamma_step {gamma_step} is not in range"
    assert 0.0 <= gamma_pulse <= 1e-3, f"gamma_pulse {gamma_pulse} is not in range"
    assert 0.0 <= gamma_step <= 1e-3, f"gamma_step {gamma_step} is not in range"
    assert 0.0 <= gamma_exog <= 20.0, f"gamma_exog {gamma_exog} is not in range"


def test_phase1_ads_have_no_effect_phase1_json():
    # Use shared scenario to build the series that ignores exogenous effect
    total = scenario_ads_no_effect()
    idx = total.index
    plot_df = pd.DataFrame(index=idx)

    # Rebuild exogenous features aligned to the scenario's index
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)

    bkps = detect_change_points(total, max_changes=4, min_seg_len=3, return_mode="indices")

    fit = fit_piecewise_logistic(total_series=total, breakpoints=bkps, events_df=None, extra_exog=exog)

    carrying_capacity = fit.carrying_capacity
    gamma_intercept = fit.gamma_intercept
    gamma_pulse = fit.gamma_pulse
    gamma_step = fit.gamma_step
    gamma_exog = fit.gamma_exog

    assert 100 <= carrying_capacity <= 30000, f"carrying_capacity {carrying_capacity} is not in range"
    assert -500.0 <= gamma_intercept <= 500.0, f"gamma_intercept {gamma_intercept} is not in range"
    assert 0.0 <= gamma_pulse <= 1e-3, f"gamma_pulse {gamma_pulse} is not in range"
    assert 0.0 <= gamma_step <= 1e-3, f"gamma_step {gamma_step} is not in range"
    assert 0.0 <= gamma_exog <= 20.0, f"gamma_exog {gamma_exog} is not in range"
