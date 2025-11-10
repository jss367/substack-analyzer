"""
Reusable scenario definitions for calibration tests and plotting tools.
"""

import pandas as pd

from substack_analyzer.analysis import build_events_features
from substack_analyzer.utils_for_tests import ad_spend_csv_with_spikes, synthesize_series_with_exog


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


def top_tier_sustained_marketing_series() -> pd.Series:
    params = scenario_top_tier_sustained_marketing()
    idx = pd.period_range("2020-01", periods=params["months"], freq="M").to_timestamp("M")
    vals = [
        15000,
        15600,
        16300,
        17100,
        18000,
        19000,
        20100,
        21300,
        22600,
        24000,
        25500,
        27100,
        28800,
        30600,
        32500,
        34500,
        36600,
        38800,
        41100,
        43500,
        46000,
        48600,
        51300,
        54100,
        57000,
        60000,
        63100,
        66300,
        69600,
        73000,
        76500,
        80100,
        83800,
        87600,
        91500,
        95500,
        99600,
        103800,
        108100,
        112500,
        117000,
        121600,
        126300,
        131100,
        136000,
        141000,
        146100,
        151300,
    ]
    return pd.Series(vals, index=idx)


def small_breakout_series() -> pd.Series:
    params = scenario_small_breakout()
    idx = pd.period_range("2020-01", periods=params["months"], freq="M").to_timestamp("M")
    vals = [
        300,
        305,
        310,
        316,
        322,
        329,
        336,
        344,
        352,
        361,
        370,
        380,
        391,
        403,
        416,
        430,
        445,
        461,
        478,
        496,
        515,
        535,
        556,
        578,
        601,
        625,
        650,
        676,
        703,
        731,
        800,
        880,
        970,
        1070,
        1180,
        1300,
        1430,
        1570,
        1880,
        2050,
        2230,
        2420,
    ]
    return pd.Series(vals, index=idx)


def niche_steady_series() -> pd.Series:
    params = scenario_niche_steady()
    idx = pd.period_range("2020-01", periods=params["months"], freq="M").to_timestamp("M")
    vals = [
        800,
        812,
        824,
        836,
        848,
        861,
        874,
        868,
        872,
        895,
        905,
        898,
        912,
        927,
        941,
        936,
        960,
        965,
        980,
        996,
        1011,
        1006,
        1022,
        1045,
        1055,
        1070,
        1083,
        1101,
        1115,
        1149,
        1146,
        1162,
        1170,
        1193,
        1210,
        1225,
    ]
    return pd.Series(vals, index=idx)


def mid_sized_seasonal_conference_series() -> pd.Series:
    params = scenario_mid_sized_seasonal_conference()
    idx = pd.period_range("2020-01", periods=params["months"], freq="M").to_timestamp("M")
    vals = [
        2500,
        2550,
        2600,
        2660,
        2720,
        2790,
        2870,
        2960,
        3060,
        3170,
        3290,
        3420,
        3560,
        3710,
        3870,
        4040,
        4220,
        4410,
        4610,
        4820,
        5040,
        5270,
        5510,
        5760,
        6020,
        6290,
        6570,
        6860,
        7160,
        7470,
    ]
    return pd.Series(vals, index=idx)


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
    exog_series = pd.Series([0.0, pd.NA, 1.0, 0.0, 1.0, 0.0, 1.0], index=idx)
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


def test_phase1_ads_really_valuable_phase1_json():
    # Monthly timeline
    idx = pd.period_range("2022-01", periods=36, freq="M").to_timestamp("M")
    plot_df = pd.DataFrame(index=idx)

    # Ad spend file (constant spend) -> features with ad_effect_log
    spikes = {
        idx[6]: 3000.0,  # mid-year push
        idx[18]: 2000.0,  # another big campaign
    }
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)

    # Build Total series that actually uses exogenous effect (positive influence)
    total = synthesize_series_with_exog(idx, K=20000.0, r=0.03, exog=exog, g_exog=0)

    return total


def test_phase1_ads_extremely_valuable_phase1_json():
    """
    Scenario: almost no organic growth without ads, but enormous growth driven by ads.

    Implementation details:
    - Organic growth rate r is set very low (≈0.001), so logistic term alone barely moves.
    - Constant monthly ad spend is applied to generate a sustained exogenous driver.
    - Exogenous gain g_exog is set ~10x higher than a "typical" strong value to create
      an outsized ad-driven effect.
    """
    # Monthly timeline
    idx = pd.period_range("2022-01", periods=36, freq="M").to_timestamp("M")
    plot_df = pd.DataFrame(index=idx)

    # Spiky ad spend: exactly two large campaigns
    spikes = {
        idx[6]: 15000.0,
        idx[18]: 20000.0,
    }
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float)

    # Build Total series: negligible organic (r very small), extreme exogenous gain
    total = synthesize_series_with_exog(idx, K=20000.0, r=0.001, exog=exog, g_exog=1000.0)

    return total


def test_phase1_ads_have_no_effect_phase1_json():
    # Monthly timeline
    idx = pd.period_range("2022-01", periods=36, freq="M").to_timestamp("M")
    plot_df = pd.DataFrame(index=idx)

    # Ad spend present, but the synthesized series ignores exogenous effect (g_exog=0)
    spikes = {
        idx[6]: 3000.0,  # mid-year push
        idx[18]: 2000.0,  # another big campaign
    }
    ad_file = ad_spend_csv_with_spikes(idx, spikes)
    _covariates_df, features_df = build_events_features(plot_df, ad_file=ad_file)
    exog = features_df["ad_effect_log"].astype(float) * 0.0

    # Build Total series without exogenous effect
    total = synthesize_series_with_exog(idx, K=20000.0, r=0.15, exog=exog, g_exog=0.0)

    return total
