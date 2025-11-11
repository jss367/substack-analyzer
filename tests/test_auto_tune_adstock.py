import pandas as pd

from substack_analyzer.analysis import auto_tune_adstock
from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.scenarios import (
    scenario_ads_extremely_valuable,
    scenario_ads_no_effect,
    scenario_ads_really_valuable,
)
from substack_analyzer.utils_for_tests import ad_spend_csv_with_spikes


def test_auto_tune_adstock_really_valuable():
    # Series synthesized with a strong exogenous ad-effect signal
    total = scenario_ads_really_valuable()
    idx = total.index
    plot_df = pd.DataFrame(index=idx)

    # Ad spend used to construct the scenario's exogenous signal
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)

    lam_best, theta_best, cov_df, feat_df = auto_tune_adstock(
        plot_df,
        ad_file=ad_file,
        breakpoints=[],
        events_df=None,
        fit_series=total,
    )

    # Returned parameters should come from the default grids
    assert lam_best in {0.0, 0.2, 0.4, 0.6, 0.8, 0.9}
    assert theta_best in {100.0, 250.0, 500.0, 1000.0, 2000.0}

    # Using the tuned features should yield a very strong fit with exogenous regressor
    exog = feat_df["ad_effect_log"].astype(float)
    fit = fit_piecewise_logistic(total_series=total, breakpoints=[], events_df=None, extra_exog=exog)
    assert len(fit.fitted_series) == len(total)
    assert fit.r2_on_deltas > 0.95
    assert fit.sse <= 100


def test_auto_tune_adstock_extremely_valuable():
    # Series synthesized with negligible organic growth and extreme ad-driven effect
    total = scenario_ads_extremely_valuable()
    idx = total.index
    plot_df = pd.DataFrame(index=idx)

    # Ad spend used to construct the scenario's exogenous signal
    spikes = {idx[6]: 15000.0, idx[18]: 20000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)

    lam_best, theta_best, cov_df, feat_df = auto_tune_adstock(
        plot_df,
        ad_file=ad_file,
        breakpoints=[],
        events_df=None,
        fit_series=total,
    )

    assert lam_best in {0.0, 0.2, 0.4, 0.6, 0.8, 0.9}
    assert theta_best in {100.0, 250.0, 500.0, 1000.0, 2000.0}

    exog = feat_df["ad_effect_log"].astype(float)
    fit = fit_piecewise_logistic(total_series=total, breakpoints=[], events_df=None, extra_exog=exog)
    assert len(fit.fitted_series) == len(total)
    # This scenario is designed to be dominated by exogenous effect
    assert fit.r2_on_deltas > 0.9
    assert fit.sse <= 4_200_000


def test_auto_tune_adstock_ads_have_no_effect():
    # Series synthesized to ignore exogenous effect
    total = scenario_ads_no_effect()
    idx = total.index
    plot_df = pd.DataFrame(index=idx)

    # Still provide an ad spend file; tuned exog should not materially help/hurt
    spikes = {idx[6]: 3000.0, idx[18]: 2000.0}
    ad_file = ad_spend_csv_with_spikes(idx, spikes)

    lam_best, theta_best, cov_df, feat_df = auto_tune_adstock(
        plot_df,
        ad_file=ad_file,
        breakpoints=[],
        events_df=None,
        fit_series=total,
    )

    exog = feat_df["ad_effect_log"].astype(float)
    fit_with_exog = fit_piecewise_logistic(total_series=total, breakpoints=[], events_df=None, extra_exog=exog)
    # fit_baseline = fit_piecewise_logistic(total_series=total, breakpoints=[], events_df=None, extra_exog=None)
    assert len(fit_with_exog.fitted_series) == len(total)
    # Auto-tuned exogenous regressor should not materially worsen the fit
    assert fit_with_exog.sse <= fit_baseline.sse * 1.05
    assert fit_with_exog.r2_on_deltas >= fit_baseline.r2_on_deltas
