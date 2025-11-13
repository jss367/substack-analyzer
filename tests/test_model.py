from substack_analyzer.model import simulate_growth
from substack_analyzer.types import AdSpendSchedule, SimulationInputs


def test_simulate_growth_basic():
    inputs = SimulationInputs(
        starting_free_subscribers=1000,
        starting_premium_subscribers=100,
        horizon_months=12,
        organic_monthly_growth_rate=0.01,
        monthly_churn_rate_free=0.0,
        monthly_churn_rate_premium=0.0,
        new_subscriber_premium_conv_rate=0.02,
        ongoing_premium_conv_rate=0.0003,
        cost_per_new_free_subscriber=2.0,
        ad_spend_schedule=AdSpendSchedule.constant(0.0),
        ad_manager_monthly_fee=0.0,
        premium_monthly_price_gross=10.0,
        substack_fee_pct=0.10,
        stripe_fee_pct=0.036,
        stripe_flat_fee=0.30,
        annual_share=0.0,
        premium_annual_price_gross=70.0,
    )
    result = simulate_growth(inputs)
    df = result.monthly
    assert len(df) == 12
    assert df["total_subscribers"].iloc[-1] >= df["total_subscribers"].iloc[0]
    # Net revenue should be non-negative given no ad costs
    assert (df["net_revenue"] >= 0).all()


def test_carrying_capacity_replaces_churn_near_ceiling():
    inputs = SimulationInputs(
        starting_free_subscribers=8000,
        starting_premium_subscribers=1500,
        carrying_capacity=10000,
        horizon_months=36,
        organic_monthly_growth_rate=0.06,
        monthly_churn_rate_free=0.02,
        monthly_churn_rate_premium=0.02,
        new_subscriber_premium_conv_rate=0.01,
        ongoing_premium_conv_rate=0.0002,
        cost_per_new_free_subscriber=2.0,
        ad_spend_schedule=AdSpendSchedule.constant(0.0),
        ad_manager_monthly_fee=0.0,
        premium_monthly_price_gross=10.0,
        substack_fee_pct=0.10,
        stripe_fee_pct=0.036,
        stripe_flat_fee=0.30,
        annual_share=0.0,
        premium_annual_price_gross=70.0,
    )

    result = simulate_growth(inputs)
    df = result.monthly

    tail = df["total_subscribers"].iloc[-6:]
    assert tail.iloc[-1] >= tail.iloc[0]


# end
