"""
Reusable scenario definitions for calibration tests and plotting tools.
"""

import pandas as pd


def scenario_top_tier_sustained_marketing() -> dict:
    """
    Top-tier newsletter saturating after sustained marketing.
    """

    def case1_events(idx: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [idx[14], idx[26]],
                "type": ["Brand Campaign", "Referral push"],
                "persistence": ["persistent", "transient"],
                "cost": [1.0, 1.0],
            }
        )

    return {
        "description": "top-tier newsletter saturating after sustained marketing",
        "start": 15_000.0,
        "months": 48,
        "breakpoints": [12, 24, 36],
        "carrying_capacity": 500_000.0,
        "segment_rates": [0.45, 0.30, 0.18, 0.12],
        "gamma_pulse": 3_500.0,
        "gamma_step": 1_200.0,
        "events": case1_events,
    }


def scenario_small_breakout() -> dict:
    """
    Small newsletter that eventually breaks out.
    """

    def case2_events(idx: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [idx[24]],
                "type": ["Product Launch"],
                "persistence": ["persistent"],
                "cost": [1.0],
            }
        )

    return {
        "description": "small newsletter that eventually breaks out",
        "start": 300.0,
        "months": 42,
        "breakpoints": [18, 30],
        "carrying_capacity": 90000.0,
        "segment_rates": [0.05, 0.10, 0.26],
        "gamma_pulse": 0.0,
        "gamma_step": 80.0,
        "events": case2_events,
    }


def scenario_niche_steady() -> dict:
    """
    Niche newsletter that stays relatively small but steady.
    """

    def case3_events(idx: pd.DatetimeIndex):
        return None

    return {
        "description": "niche newsletter that stays relatively small but steady",
        "start": 800.0,
        "months": 36,
        "breakpoints": [],
        "carrying_capacity": 4_000.0,
        "segment_rates": [0.04],
        "gamma_pulse": 0.0,
        "gamma_step": 0.0,
        "events": case3_events,
    }


def scenario_mid_sized_seasonal_conference() -> dict:
    """
    Mid-sized publication with seasonal campaigns and a big conference push.
    """

    def case4_events(idx: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [idx[5], idx[11], idx[17], idx[23]],
                "type": ["Guest Post", "Holiday Promo", "Guest Post", "Conference"],
                "persistence": ["transient", "transient", "transient", "persistent"],
                "cost": [1.0, 1.5, 1.0, 1.0],
            }
        )

    return {
        "description": "mid-sized publication with seasonal campaigns and a big conference push",
        "start": 2_500.0,
        "months": 30,
        "breakpoints": [10, 20],
        "carrying_capacity": 75_000.0,
        "segment_rates": [0.07, 0.09, 0.12],
        "gamma_pulse": 420.0,
        "gamma_step": 260.0,
        "events": case4_events,
    }


def realistic_growth_profiles_cases() -> list[dict]:
    """
    Backwards-compatible list of the four individual cases.
    """
    return [
        scenario_top_tier_sustained_marketing(),
        scenario_small_breakout(),
        scenario_niche_steady(),
        scenario_mid_sized_seasonal_conference(),
    ]


def gm_series_values() -> pd.Series:
    """
    Fixed 'gm' series used in tests; breakpoints are typically detected automatically.
    """
    vals = [
        0,
        0,
        0,
        2,
        3,
        4,
        4,
        4,
        4,
        5,
        7,
        30,
        31,
        31,
        32,
        33,
        33,
        33,
        33,
        35,
        36,
        36,
        35,
        36,
        39,
        42,
        42,
        44,
        45,
        45,
        47,
        50,
        56,
        60,
        82,
        93,
        104,
        109,
        108,
        116,
        121,
        124,
        128,
        128,
        131,
        134,
        133,
        134,
    ]
    idx = pd.period_range("2021-10", periods=len(vals), freq="M").to_timestamp("M")
    return pd.Series(vals, index=idx)


def cy_series_values() -> tuple[pd.Series, list[int]]:
    """
    Series with a jump in growth rate around 2025-01; test uses a known breakpoint.
    """
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
    series = pd.Series(vals, index=idx)
    return series, [16]


def one_time_spike_series_and_events() -> tuple[pd.Series, pd.DataFrame]:
    """
    Simple series with a one-time spike explained by a transient event.
    """
    idx = pd.period_range("2024-01", periods=7, freq="M").to_timestamp("M")
    s = pd.Series([100, 100, 120, 120, 120, 120, 120], index=idx)
    events_df = pd.DataFrame({"date": [idx[2]], "type": ["promo"], "persistence": ["Transient"]})
    return s, events_df


def exog_with_nans_series_and_exog() -> tuple[pd.Series, pd.Series]:
    """
    Series whose deltas are driven by an exogenous signal that includes NaNs.
    """
    idx = pd.period_range("2024-01", periods=8, freq="M").to_timestamp("M")
    exog_deltas = [0, 2, 0, 2, 0, 2, 0]
    s_vals = [100]
    for d in exog_deltas:
        s_vals.append(s_vals[-1] + d)
    s = pd.Series(s_vals, index=idx)
    exog_series = pd.Series([0.0, 1.0, pd.NA, 1.0, 0.0, 1.0, 0.0], index=idx[1:], dtype="Float64")
    return s, exog_series.astype(float)


def three_breaks_mixed_persistence_params() -> dict:
    """
    Parameters and events for a synthetic series with three breakpoints and mixed persistence.
    Use with `fitted_series_from_params` to build the actual series.
    """
    idx = pd.period_range("2022-01", periods=42, freq="M").to_timestamp("M")
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
    return {
        "base_series": base_series,
        "breakpoints": breakpoints,
        "carrying_capacity": 150.0,
        "segment_growth_rates": segment_growth_rates,
        "events_df": events_df,
        "gamma_pulse": 3.0,
        "gamma_step": 0.6,
    }


def two_segment_ordered_series() -> tuple[pd.Series, list[int]]:
    """
    Build a series with faster growth early, slower later, and a known breakpoint at 5.
    """
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
    return s, [5]
