from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

# Default organic monthly growth rate used across the app when no estimate is available
DEFAULT_GROWTH_RATE: float = 0.10


@dataclass(frozen=True)
class PiecewiseLogisticFit:
    """
    A fit of the piecewise logistic model.
    """

    carrying_capacity: float
    segment_growth_rates: list[float]
    segment_intercepts: list[float]
    breakpoints: list[int]
    gamma_pulse: float
    gamma_step: float
    fitted_series: pd.Series
    residuals: pd.Series
    sse: float
    r2_on_deltas: float
    gamma_exog: float | None = None
    gamma_intercept: float = 0.0
    exog_lag: int | None = None
    # Optional per-segment parameters (aligned to `segment_growth_rates` order)
    segment_carrying_capacities: list[float] | None = None
    segment_gamma_pulse: list[float] | None = None
    segment_gamma_step: list[float] | None = None
    segment_gamma_exog: list[float] | None = None


@dataclass(frozen=True)
class SegmentSlope:
    start_index: int
    end_index: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    slope_per_month: float


@dataclass(frozen=True)
class AdSpendSchedule:
    """Defines monthly ad spend as a function of month index (starting at 0)."""

    get_spend_for_month: Callable[[int], float]

    @staticmethod
    def constant(monthly_spend: float) -> "AdSpendSchedule":
        return AdSpendSchedule(lambda _m: float(monthly_spend))

    @staticmethod
    def two_stage(year_1: float, year_2: float) -> "AdSpendSchedule":
        def _spend(month_index: int) -> float:
            # Months are 0-indexed. Year 1: months 0..11, Year 2 and beyond: months 12+
            if month_index < 12:
                return float(year_1)
            return float(year_2)

        return AdSpendSchedule(_spend)

    @staticmethod
    def one_time(amount: float, at_month_index: int = 0) -> "AdSpendSchedule":
        """Spend only on a single month (0-indexed)."""

        def _spend(month_index: int) -> float:
            return float(amount) if int(month_index) == int(at_month_index) else 0.0

        return AdSpendSchedule(_spend)


@dataclass(frozen=True)
class SimulationInputs:
    # Starting state
    starting_free_subscribers: int = 0
    starting_premium_subscribers: int = 0

    # Optional carrying capacities that damp growth as the audience approaches
    # the inferred ceiling from Phase 1 fits. If segment-specific capacities
    # are provided they take precedence; otherwise `carrying_capacity` is used
    # as a shared ceiling for both free and premium.
    carrying_capacity: float | None = None
    carrying_capacity_free: float | None = None
    carrying_capacity_premium: float | None = None

    # Horizon
    horizon_months: int = 60

    # Growth and churn
    organic_monthly_growth_rate: float = DEFAULT_GROWTH_RATE
    monthly_churn_rate_free: float = 0.01  # 1%
    monthly_churn_rate_premium: float = 0.01  # 1%
    monthly_downgrade_rate_premium: float = 0.0  # share of premium who step down to free

    # Conversions
    new_subscriber_premium_conv_rate: float = 0.02  # 2% of new free subs
    ongoing_premium_conv_rate: float = 0.0003  # 0.03% of existing free base per month

    # Acquisition
    cost_per_new_free_subscriber: float = 2.00
    cost_per_new_premium_subscriber: float = 10.00
    ad_spend_schedule: AdSpendSchedule = AdSpendSchedule.two_stage(3000.0, 1000.0)
    premium_ad_spend_schedule: AdSpendSchedule = AdSpendSchedule.constant(0.0)
    ad_manager_monthly_fee: float = 1500.0
    # Diminishing returns parameters for paid acquisition
    adstock_lambda: float = 0.5  # carryover of prior ad effectiveness
    ad_log_theta: float = 500.0  # scale for log(1 + adstock/theta) response curve

    # Pricing & fees (monthly plan)
    premium_monthly_price_gross: float = 10.0
    substack_fee_pct: float = 0.10  # 10%
    stripe_fee_pct: float = 0.036  # 3.6%
    stripe_flat_fee: float = 0.30  # $0.30 per transaction

    # Optional: annual pricing share (0..1). If >0, some users pay annually.
    annual_share: float = 0.0
    premium_annual_price_gross: float = 70.0


@dataclass(frozen=True)
class SimulationResult:
    monthly: pd.DataFrame

    @property
    def summary(self) -> dict[str, float]:
        last = self.monthly.iloc[-1]
        return {
            "ending_free": float(last["free_subscribers"]),
            "ending_premium": float(last["premium_subscribers"]),
            "ending_total": float(last["total_subscribers"]),
            "cumulative_net_profit": float(last["cumulative_net_profit"]),
            "cumulative_ad_spend": float(last["cumulative_ad_spend"]),
            "peak_mrr_net": float(self.monthly["mrr_net"].max()),
        }


@dataclass(frozen=True)
class EventRow:
    """Single event annotation row serialized for Phase 1 handoff."""

    date: str
    type: str
    persistence: str
    notes: str
    cost: float


@dataclass(frozen=True)
class PhaseOneOutput:
    """Portable handoff artifact from Phase 1 → Phase 2.

    All fields are JSON-serializable. Series are represented as lists of
    {"date": str, "count": float} records at month-end timestamps.
    """

    total_series: list[dict[str, Any]] | None
    paid_series: list[dict[str, Any]] | None

    breakpoints_indices: list[int]
    breakpoints_dates: list[str]

    events: list[EventRow]
    ad_spend: list[dict[str, Any]] | None

    adstock_lambda: float
    ad_log_theta: float

    detect_mode: str
    detected_target_label: str | None
    target_col_for_fit: str | None
