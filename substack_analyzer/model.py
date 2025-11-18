import numpy as np
import pandas as pd

from substack_analyzer.types import SimulationInputs, SimulationResult


def _net_monthly_revenue_per_premium(input_params: SimulationInputs) -> float:
    gross = input_params.premium_monthly_price_gross
    net = gross * (1.0 - input_params.substack_fee_pct - input_params.stripe_fee_pct) - input_params.stripe_flat_fee
    return max(net, 0.0)


def _net_annual_revenue_per_premium(input_params: SimulationInputs) -> float:
    gross = input_params.premium_annual_price_gross
    net = gross * (1.0 - input_params.substack_fee_pct - input_params.stripe_fee_pct) - input_params.stripe_flat_fee
    return max(net, 0.0)


def simulate_growth(input_params: SimulationInputs) -> SimulationResult:
    """Run a monthly simulation of subscriber and revenue dynamics.

    Model notes (MVP):
    - Free base grows via organic rate and paid acquisition (ad spend / CAC)
    - Churn applied to beginning-of-month balances
    - Premium conversions:
            - A share of this month's new free subscribers convert immediately
              (new_subscriber_premium_conv_rate)
            - A small ongoing share of existing free base converts monthly
              (ongoing_premium_conv_rate)
    - Premium churn uses monthly_churn_rate_premium
    - Revenue assumes monthly plan for all premium users unless annual_share > 0
    - Net revenue computes Substack and Stripe fees (percentage + flat)
    - Profit = net revenue - ad spend - ad manager fee
    """

    months = np.arange(input_params.horizon_months)

    columns = [
        "month",
        "free_subscribers",
        "premium_subscribers",
        "total_subscribers",
        "new_free_organic",
        "new_free_paid",
        "new_premium_paid",
        "free_churned",
        "premium_converted_from_new",
        "premium_converted_from_existing",
        "premium_churned",
        "ad_spend",
        "ad_spend_free",
        "ad_spend_premium",
        "ad_manager_fee",
        "mrr_gross",
        "mrr_net",
        "net_revenue",
        "profit",
        "cumulative_ad_spend",
        "cumulative_net_profit",
    ]
    data: list[list[float]] = []

    free_subs = float(input_params.starting_free_subscribers)
    premium_subs = float(input_params.starting_premium_subscribers)

    net_monthly = _net_monthly_revenue_per_premium(input_params)
    net_annual = _net_annual_revenue_per_premium(input_params)

    cumulative_ad_spend = 0.0
    cumulative_net_profit = 0.0
    prev_adstock_free = 0.0
    prev_adstock_premium = 0.0

    for m in months:
        # Beginning-of-month churn
        free_churned = free_subs * input_params.monthly_churn_rate_free
        premium_churned = premium_subs * input_params.monthly_churn_rate_premium

        free_subs -= free_churned
        premium_subs -= premium_churned

        # Capacity pressure: scale organic/paid acquisition as the audience
        # approaches the inferred carrying capacity (if provided). Prefer
        # segment-specific ceilings when available.
        carrying_capacity_shared = input_params.carrying_capacity or 0.0
        carrying_capacity_free = (
            carrying_capacity_shared
            if input_params.carrying_capacity_free is None
            else float(input_params.carrying_capacity_free)
        )
        carrying_capacity_premium = (
            0.0
            if input_params.carrying_capacity_premium is None
            else float(input_params.carrying_capacity_premium)
        )
        capacity_multiplier_free = 1.0
        free_capacity_denominator = free_subs
        free_capacity_gap = float("inf")
        if carrying_capacity_free > 0:
            use_shared_free = input_params.carrying_capacity_free is None and carrying_capacity_shared > 0
            free_capacity_denominator = free_subs + premium_subs if use_shared_free else free_subs
            free_capacity_gap = max(0.0, carrying_capacity_free - free_capacity_denominator)
            capacity_multiplier_free = max(0.0, 1.0 - free_capacity_denominator / carrying_capacity_free)

        # Organic growth (before capacity adjustment)
        new_free_organic_raw = free_subs * input_params.organic_monthly_growth_rate

        # Paid acquisition (before capacity adjustment)
        ad_spend_free = float(input_params.ad_spend_schedule.get_spend_for_month(m))
        ad_spend_premium = float(input_params.premium_ad_spend_schedule.get_spend_for_month(m))

        # Adstock with diminishing-returns response: log(1 + adstock/theta)
        adstock_free = ad_spend_free + input_params.adstock_lambda * prev_adstock_free
        prev_adstock_free = adstock_free

        adstock_premium = ad_spend_premium + input_params.adstock_lambda * prev_adstock_premium
        prev_adstock_premium = adstock_premium

        paid_new_free_base = (
            0.0
            if input_params.cost_per_new_free_subscriber <= 0
            else ad_spend_free / input_params.cost_per_new_free_subscriber
        )
        paid_new_premium_base = (
            0.0
            if input_params.cost_per_new_premium_subscriber <= 0
            else ad_spend_premium / input_params.cost_per_new_premium_subscriber
        )

        def _diminishing_multiplier(adstock_val: float) -> float:
            if adstock_val > 0 and input_params.ad_log_theta > 0:
                response_val = np.log1p(adstock_val / input_params.ad_log_theta)
                return response_val / (adstock_val / input_params.ad_log_theta)
            if adstock_val <= 0:
                return 0.0
            return 1.0

        paid_new_free = paid_new_free_base * _diminishing_multiplier(adstock_free)
        paid_new_premium = paid_new_premium_base * _diminishing_multiplier(adstock_premium)

        # Apply capacity adjustment while ensuring churn replacement near the ceiling
        new_free_raw_total = new_free_organic_raw + paid_new_free
        overall_multiplier = 1.0 if new_free_raw_total > 0 else 0.0
        if carrying_capacity_free > 0 and new_free_raw_total > 0:
            base_fill_free = min(new_free_raw_total, free_capacity_gap)
            excess_new_free = new_free_raw_total - base_fill_free
            adjusted_total_new_free = base_fill_free + excess_new_free * capacity_multiplier_free
            overall_multiplier = (
                adjusted_total_new_free / new_free_raw_total if new_free_raw_total > 0 else 0.0
            )

        new_free_organic = new_free_organic_raw * overall_multiplier
        new_free_paid = paid_new_free * overall_multiplier

        # Add new free
        new_free_total = new_free_organic + new_free_paid
        free_subs += new_free_total

        # Conversions to premium
        convert_from_new_raw = new_free_total * input_params.new_subscriber_premium_conv_rate
        convert_from_existing_raw = max(free_subs - new_free_total, 0.0) * input_params.ongoing_premium_conv_rate

        # Premium paid acquisition (direct to premium)
        premium_inflow_raw = convert_from_new_raw + convert_from_existing_raw + paid_new_premium
        premium_multiplier = 1.0 if premium_inflow_raw > 0 else 0.0
        if carrying_capacity_premium > 0 and premium_inflow_raw > 0:
            capacity_gap_premium = max(0.0, carrying_capacity_premium - premium_subs)
            base_fill_premium = min(premium_inflow_raw, capacity_gap_premium)
            excess_premium = premium_inflow_raw - base_fill_premium
            capacity_multiplier_premium = max(
                0.0, 1.0 - (premium_subs + base_fill_premium) / carrying_capacity_premium
            )
            adjusted_total_premium = base_fill_premium + excess_premium * capacity_multiplier_premium
            premium_multiplier = (
                adjusted_total_premium / premium_inflow_raw if premium_inflow_raw > 0 else 0.0
            )

        convert_from_new = convert_from_new_raw * premium_multiplier
        convert_from_existing = convert_from_existing_raw * premium_multiplier
        paid_new_premium_adjusted = paid_new_premium * premium_multiplier

        # Apply conversions: move from free to premium
        total_convert = convert_from_new + convert_from_existing
        free_subs = max(free_subs - total_convert, 0.0)
        premium_subs += total_convert + paid_new_premium_adjusted

        # Revenue
        # Split premium base into monthly vs annual cohorts
        monthly_premium = premium_subs * (1.0 - input_params.annual_share)
        annual_premium = premium_subs * input_params.annual_share

        mrr_gross = monthly_premium * input_params.premium_monthly_price_gross
        mrr_net = monthly_premium * net_monthly

        # Annual revenue recognized this month (simplified: evenly amortized)
        annual_revenue_net_month = (annual_premium * net_annual) / 12.0

        net_revenue = mrr_net + annual_revenue_net_month

        ad_spend_total = ad_spend_free + ad_spend_premium
        ad_manager_fee = input_params.ad_manager_monthly_fee if ad_spend_total > 0 else 0.0
        profit = net_revenue - ad_spend_total - ad_manager_fee

        cumulative_ad_spend += ad_spend_total
        cumulative_net_profit += profit

        total_subscribers = free_subs + premium_subs

        data.append(
            [
                float(m + 1),
                free_subs,
                premium_subs,
                total_subscribers,
                new_free_organic,
                new_free_paid,
                paid_new_premium_adjusted,
                free_churned,
                convert_from_new,
                convert_from_existing,
                premium_churned,
                ad_spend_total,
                ad_spend_free,
                ad_spend_premium,
                ad_manager_fee,
                mrr_gross,
                mrr_net,
                net_revenue,
                profit,
                cumulative_ad_spend,
                cumulative_net_profit,
            ]
        )

    monthly_df = pd.DataFrame(data, columns=columns)
    # Round subscriber stock columns to integers for readability
    for _col in ["free_subscribers", "premium_subscribers", "total_subscribers"]:
        if _col in monthly_df.columns:
            monthly_df[_col] = monthly_df[_col].round().astype(int)

    # Round monthly flow counts to integers
    flow_cols = [
        "new_free_organic",
        "new_free_paid",
        "free_churned",
        "premium_converted_from_new",
        "premium_converted_from_existing",
        "premium_churned",
        "new_premium_paid",
    ]
    for _col in flow_cols:
        if _col in monthly_df.columns:
            monthly_df[_col] = monthly_df[_col].round().astype(int)

    # Round monetary columns to whole dollars
    money_cols = [
        "ad_spend",
        "ad_spend_free",
        "ad_spend_premium",
        "ad_manager_fee",
        "mrr_gross",
        "mrr_net",
        "net_revenue",
        "profit",
        "cumulative_ad_spend",
        "cumulative_net_profit",
    ]
    for _col in money_cols:
        if _col in monthly_df.columns:
            monthly_df[_col] = monthly_df[_col].round().astype(int)

    # Ensure month is an integer index-like column
    if "month" in monthly_df.columns:
        monthly_df["month"] = monthly_df["month"].round().astype(int)
    return SimulationResult(monthly=monthly_df)
