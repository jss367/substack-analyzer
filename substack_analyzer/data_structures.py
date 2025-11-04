from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PiecewiseLogisticFit:
    """
    A fit of the piecewise logistic model.
    """

    carrying_capacity: float
    segment_growth_rates: list[float]
    breakpoints: list[int]
    gamma_pulse: float
    gamma_step: float
    fitted_series: pd.Series
    residuals: pd.Series
    sse: float
    r2_on_deltas: float
    gamma_exog: float | None = None
    gamma_intercept: float = 0.0
