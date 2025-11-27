from substack_analyzer.types import DualSeriesFit, PiecewiseLogisticFit


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
