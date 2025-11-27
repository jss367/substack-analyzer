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
