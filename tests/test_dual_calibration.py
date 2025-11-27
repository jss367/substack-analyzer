import pandas as pd
import numpy as np
from substack_analyzer.types import DualSeriesFit, PiecewiseLogisticFit
from substack_analyzer.calibration import fit_dual_series


def test_dual_series_fit_dataclass_exists():
    """DualSeriesFit should hold two PiecewiseLogisticFit objects and inferred parameters."""
    # Arrange: Create minimal mock fits
    free_fit = PiecewiseLogisticFit(
        carrying_capacity=10000.0,
        segment_growth_rates=[0.15],
        segment_intercepts=[10.0],
        breakpoints=[],
        gamma_pulse=0.0,
        gamma_step=0.0,
        fitted_series=None,  # type: ignore
        residuals=None,  # type: ignore
        sse=100.0,
        r2_on_deltas=0.85,
    )
    premium_fit = PiecewiseLogisticFit(
        carrying_capacity=1000.0,
        segment_growth_rates=[0.08],
        segment_intercepts=[2.0],
        breakpoints=[],
        gamma_pulse=0.0,
        gamma_step=0.0,
        fitted_series=None,  # type: ignore
        residuals=None,  # type: ignore
        sse=50.0,
        r2_on_deltas=0.90,
    )

    # Act: Create DualSeriesFit
    dual_fit = DualSeriesFit(
        free_fit=free_fit,
        premium_fit=premium_fit,
        inferred_conversion_rate=0.05,
        inferred_churn_rate_free=0.02,
        inferred_churn_rate_premium=0.01,
    )

    # Assert: Should have both fits and inferred params
    assert dual_fit.free_fit == free_fit
    assert dual_fit.premium_fit == premium_fit
    assert dual_fit.inferred_conversion_rate == 0.05
    assert dual_fit.inferred_churn_rate_free == 0.02
    assert dual_fit.inferred_churn_rate_premium == 0.01


def test_fit_dual_series_fits_both_independently():
    """fit_dual_series should fit free and premium series separately."""
    # Arrange: Create synthetic free and premium series
    dates = pd.date_range("2023-01-31", periods=24, freq="ME")

    # Free: starts at 100, grows to ~1000
    free_values = 100 * (1.15 ** np.arange(24))
    free_series = pd.Series(free_values, index=dates, name="free")

    # Premium: starts at 10, grows to ~100 (slower growth)
    premium_values = 10 * (1.08 ** np.arange(24))
    premium_series = pd.Series(premium_values, index=dates, name="premium")

    # Act: Fit both series
    dual_fit = fit_dual_series(
        free_series=free_series,
        premium_series=premium_series,
        breakpoints=[],
    )

    # Assert: Should have both fits
    assert dual_fit.free_fit is not None
    assert dual_fit.premium_fit is not None
    assert dual_fit.free_fit.carrying_capacity > 0
    assert dual_fit.premium_fit.carrying_capacity > 0
    assert len(dual_fit.free_fit.segment_growth_rates) > 0
    assert len(dual_fit.premium_fit.segment_growth_rates) > 0


def test_fit_dual_series_infers_conversion_rate():
    """fit_dual_series should infer conversion rate from growth relationship."""
    # Arrange: Free grows faster, premium grows from conversions
    dates = pd.date_range("2023-01-31", periods=24, freq="ME")

    # Simulate: Free grows 10% monthly, 5% convert to premium
    free_base = 1000.0
    free_values = []
    premium_values = []
    premium_base = 50.0

    for month in range(24):
        free_values.append(free_base)
        premium_values.append(premium_base)

        # Free growth: 10% organic growth
        new_free = free_base * 0.10
        # Conversions: 5% of free base converts to premium
        conversions = free_base * 0.05

        free_base += new_free - conversions
        premium_base += conversions

    free_series = pd.Series(free_values, index=dates, name="free")
    premium_series = pd.Series(premium_values, index=dates, name="premium")

    # Act: Fit and infer
    dual_fit = fit_dual_series(
        free_series=free_series,
        premium_series=premium_series,
        breakpoints=[],
    )

    # Assert: Should infer a conversion rate around 5%
    assert dual_fit.inferred_conversion_rate is not None
    # Allow generous tolerance since fitting may not be perfect
    assert 0.01 <= dual_fit.inferred_conversion_rate <= 0.15


def test_fit_dual_series_infers_churn_rates():
    """fit_dual_series should infer different churn rates for free vs premium."""
    # Arrange: Series with different churn characteristics
    dates = pd.date_range("2023-01-31", periods=24, freq="ME")

    # Free: high churn (5% monthly), premium: low churn (1% monthly)
    free_base = 1000.0
    premium_base = 200.0
    free_values = []
    premium_values = []

    for month in range(24):
        free_values.append(free_base)
        premium_values.append(premium_base)

        # Growth minus churn
        free_base = free_base * 1.10 * 0.95  # +10% growth, -5% churn
        premium_base = premium_base * 1.05 * 0.99  # +5% growth, -1% churn

    free_series = pd.Series(free_values, index=dates, name="free")
    premium_series = pd.Series(premium_values, index=dates, name="premium")

    # Act
    dual_fit = fit_dual_series(
        free_series=free_series,
        premium_series=premium_series,
        breakpoints=[],
    )

    # Assert: Free churn should be higher than premium churn
    assert dual_fit.inferred_churn_rate_free is not None
    assert dual_fit.inferred_churn_rate_premium is not None
    assert dual_fit.inferred_churn_rate_free > dual_fit.inferred_churn_rate_premium
