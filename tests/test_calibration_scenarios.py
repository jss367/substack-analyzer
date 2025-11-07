import numpy as np
import pandas as pd

from substack_analyzer.calibration import fit_piecewise_logistic, fitted_series_from_params
from substack_analyzer.scenarios import (
    scenario_mid_sized_seasonal_conference,
    scenario_niche_steady,
    scenario_small_breakout,
    scenario_top_tier_sustained_marketing,
)


def _run_case_and_assert(case: dict):
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
    thresholds = {
        "top-tier newsletter saturating after sustained marketing": 1e-8,
        "small newsletter that eventually breaks out": 1e-10,
        "niche newsletter that stays relatively small but steady": 1e-10,
        "mid-sized publication with seasonal campaigns and a big conference push": 1e-8,
    }
    assert fit.sse < thresholds[case["description"]], f"{case['description']} SSE too high: {fit.sse}"
    assert np.isclose(fit.carrying_capacity, case["carrying_capacity"], rtol=1e-6, atol=1e-6)
    assert np.allclose(fit.segment_growth_rates, case["segment_rates"], rtol=1e-6, atol=1e-6)
    assert np.isclose(fit.gamma_pulse, case["gamma_pulse"], atol=1e-6)
    assert np.isclose(fit.gamma_step, case["gamma_step"], atol=1e-6)
    assert fit.r2_on_deltas > 0.999999


def test_scenario_top_tier_sustained_marketing():
    _run_case_and_assert(scenario_top_tier_sustained_marketing())


def test_scenario_small_breakout():
    _run_case_and_assert(scenario_small_breakout())


def test_scenario_niche_steady():
    _run_case_and_assert(scenario_niche_steady())


def test_scenario_mid_sized_seasonal_conference():
    _run_case_and_assert(scenario_mid_sized_seasonal_conference())
