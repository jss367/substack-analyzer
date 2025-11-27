"""Integration test for dual-series calibration workflow."""
import pandas as pd
import numpy as np
from substack_analyzer.calibration import fit_dual_series
from substack_analyzer.model import simulate_growth
from substack_analyzer.types import SimulationInputs


def test_dual_fit_to_simulation_workflow():
    """Test complete workflow: load data → dual fit → use inferred params in simulation."""
    # Arrange: Create realistic free and premium series
    dates = pd.date_range("2023-01-31", periods=24, freq="ME")

    # Free: grows from 1000 to ~5000 with 8% monthly growth
    free_values = 1000 * (1.08 ** np.arange(24))
    free_series = pd.Series(free_values, index=dates, name="free")

    # Premium: grows from 100 to ~400 (5% monthly growth from conversions)
    premium_values = 100 * (1.05 ** np.arange(24))
    premium_series = pd.Series(premium_values, index=dates, name="premium")

    # Act 1: Fit dual series
    dual_fit = fit_dual_series(
        free_series=free_series,
        premium_series=premium_series,
        breakpoints=[],
    )

    # Assert 1: Inferred parameters are reasonable
    assert dual_fit.inferred_conversion_rate is not None
    assert 0.0 < dual_fit.inferred_conversion_rate < 0.2
    assert dual_fit.inferred_churn_rate_free is not None
    assert dual_fit.inferred_churn_rate_premium is not None

    # Act 2: Use inferred parameters in simulation
    sim_inputs = SimulationInputs(
        starting_free_subscribers=int(free_series.iloc[-1]),
        starting_premium_subscribers=int(premium_series.iloc[-1]),
        horizon_months=12,
        organic_monthly_growth_rate=dual_fit.free_fit.segment_growth_rates[-1],
        monthly_churn_rate_free=dual_fit.inferred_churn_rate_free,
        monthly_churn_rate_premium=dual_fit.inferred_churn_rate_premium,
        new_subscriber_premium_conv_rate=dual_fit.inferred_conversion_rate,
        carrying_capacity_free=dual_fit.free_fit.carrying_capacity,
        carrying_capacity_premium=dual_fit.premium_fit.carrying_capacity,
        premium_monthly_price_gross=10.0,
    )

    result = simulate_growth(sim_inputs)

    # Assert 2: Simulation runs successfully
    assert result is not None
    assert len(result.monthly) == 12
    assert result.monthly["free_subscribers"].iloc[-1] > 0
    assert result.monthly["premium_subscribers"].iloc[-1] > 0
    assert result.monthly["net_revenue"].iloc[-1] > 0
