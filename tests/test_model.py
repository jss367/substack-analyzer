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


def test_paid_acquisition_has_diminishing_returns_with_adstock():
    inputs = SimulationInputs(
        starting_free_subscribers=0,
        starting_premium_subscribers=0,
        carrying_capacity=None,
        horizon_months=3,
        organic_monthly_growth_rate=0.0,
        monthly_churn_rate_free=0.0,
        monthly_churn_rate_premium=0.0,
        new_subscriber_premium_conv_rate=0.0,
        ongoing_premium_conv_rate=0.0,
        cost_per_new_free_subscriber=1.0,
        ad_spend_schedule=AdSpendSchedule.constant(10000.0),
        ad_manager_monthly_fee=0.0,
        adstock_lambda=0.5,
        ad_log_theta=1000.0,
    )

    result = simulate_growth(inputs)
    df = result.monthly

    # With adstock + log response, each successive month should yield fewer paid new users
    paid_new = df["new_free_paid"].tolist()
    assert paid_new[0] > paid_new[1] > paid_new[2]
    # And the effective acquisitions are far below the naive linear ad_spend/CAC assumption
    assert max(paid_new) < 10000


def test_premium_paid_acquisition_uses_separate_cac():
    inputs = SimulationInputs(
        starting_free_subscribers=0,
        starting_premium_subscribers=0,
        horizon_months=1,
        organic_monthly_growth_rate=0.0,
        monthly_churn_rate_free=0.0,
        monthly_churn_rate_premium=0.0,
        new_subscriber_premium_conv_rate=0.0,
        ongoing_premium_conv_rate=0.0,
        cost_per_new_free_subscriber=1_000_000.0,
        cost_per_new_premium_subscriber=10.0,
        ad_spend_schedule=AdSpendSchedule.constant(0.0),
        premium_ad_spend_schedule=AdSpendSchedule.constant(1000.0),
        ad_manager_monthly_fee=0.0,
        adstock_lambda=0.0,
        ad_log_theta=1.0,
    )

    result = simulate_growth(inputs)
    df = result.monthly
    assert int(df["new_free_paid"].iloc[0]) == 0
    assert df["new_premium_paid"].iloc[0] > 0
    assert df["ad_spend"].iloc[0] == 1000
    assert df["ad_spend_premium"].iloc[0] == 1000


def test_carrying_capacities_are_segment_specific():
    inputs = SimulationInputs(
        starting_free_subscribers=500,
        starting_premium_subscribers=0,
        carrying_capacity_free=1_000.0,
        carrying_capacity_premium=50.0,
        horizon_months=6,
        organic_monthly_growth_rate=0.5,
        monthly_churn_rate_free=0.0,
        monthly_churn_rate_premium=0.0,
        new_subscriber_premium_conv_rate=0.5,
        ongoing_premium_conv_rate=0.0,
        cost_per_new_free_subscriber=0.0,
        ad_spend_schedule=AdSpendSchedule.constant(0.0),
        ad_manager_monthly_fee=0.0,
    )

    result = simulate_growth(inputs)
    df = result.monthly
    # Premium subscribers should be capped by the premium capacity
    assert df["premium_subscribers"].max() <= 50
    # Free base should still be able to grow toward its own ceiling
    assert df["free_subscribers"].max() > 900


# end
