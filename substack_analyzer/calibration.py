from typing import Sequence

import numpy as np
import pandas as pd

from substack_analyzer.types import PiecewiseLogisticFit
from substack_analyzer.utils import ensure_month_end_index


def _segments_from_breakpoints(n: int, breakpoints: Sequence[int]) -> list[tuple[int, int]]:
    if not breakpoints:
        return [(0, n - 1)]
    segments: list[tuple[int, int]] = []
    start = 0
    for bp in breakpoints:
        end = max(min(bp - 1, n - 2), start)  # end applies to delta index; safe bound
        if end >= start:
            segments.append((start, end))
        start = bp
    if start <= n - 2:
        segments.append((start, n - 2))
    return segments


def _event_regressors(index: pd.DatetimeIndex, events_df: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build pulse and step event regressors aligned to the deltas index.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Month-end index corresponding to ΔS_t rows (i.e., original series index[1:]).
    events_df : pd.DataFrame | None
        Optional events table. Expected columns:
        - 'date': event month (any parseable datetime); normalized to month-end
        - 'persistence': one of {'persistent', 'transient', 'no effect'} (case-insensitive)
        - 'cost' (optional): numeric weight applied to the event (defaults to 1.0)

    Behavior
    --------
    - Dates are coerced to month-end.
    - 'persistent': contributes to a step regressor from event month onward.
    - 'transient': contributes to a pulse regressor at the event month only.
    - 'no effect' or missing/invalid rows are ignored.
    - 'cost' scales the magnitude of the contribution when present.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (pulse, step), each length len(index), dtype float.
    """
    if events_df is None or events_df.empty:
        return np.zeros(len(index)), np.zeros(len(index))
    df = events_df.dropna(subset=["date"]).copy()
    if df.empty:
        return np.zeros(len(index)), np.zeros(len(index))
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp("M")
    pulse = np.zeros(len(index), dtype=float)
    step = np.zeros(len(index), dtype=float)
    for _, row in df.iterrows():
        when: pd.Timestamp = row["date"]
        # Pulse/step weighting: use cost as weight for any pulse occurrence when provided
        weight = float(row.get("cost", 1.0) or 1.0)
        persistence = str(row.get("persistence", "")).strip().lower()
        if when in index:
            i = int(index.get_loc(when))
            if persistence == "no effect":
                continue
            if persistence == "persistent":
                step[index >= when] += weight
            elif persistence == "transient":
                # Pulse at the event month (always weight by cost if available)
                pulse[i] += weight
    return pulse, step


def fit_piecewise_logistic(
    total_series: pd.Series,
    breakpoints: list[int],
    events_df: pd.DataFrame | None = None,
    k_grid: Sequence[float] | None = None,
    extra_exog: pd.Series | None = None,
    exog_lags: Sequence[int] | None = None,
) -> PiecewiseLogisticFit:
    """Fit a piecewise-logistic model on monthly totals via grid-search over K and OLS.

    When ``extra_exog`` is provided, the fitter will, by default, evaluate both contemporaneous
    alignment and a one-period lag (``exog_lags`` defaults to ``[0, 1]``) so that callers that
    constructed their feature on the total-series index do not need to manually shift it to match
    the ΔS index.
    """
    input_series = ensure_month_end_index(total_series)
    if input_series.size < 4:
        raise ValueError("Need at least 4 months of data to fit the model")
    # Construct deltas and base regressor X_t(K)
    y = input_series.diff().dropna()
    s_lag = input_series.shift(1).reindex(y.index).astype(float)
    n = y.size

    # Build segments on the index of y (which starts at original index[1])
    # Sanitize breakpoints relative to original series length (len(input_series))
    n_series = len(input_series)
    bps = sorted({int(b) for b in breakpoints if 1 <= int(b) <= n_series - 2}) if breakpoints else []
    seg_bounds = _segments_from_breakpoints(n_series, bps)
    num_segments = len(seg_bounds)

    # Events
    pulse, step = _event_regressors(y.index, events_df)
    pulse = np.asarray(pd.Series(pulse, index=y.index), dtype=float)
    step = np.asarray(pd.Series(step, index=y.index), dtype=float)

    # Optional exogenous aligned to y index (support trying simple integer lags)
    exog_candidates: list[tuple[int | None, np.ndarray | None]] = [(None, None)]
    if extra_exog is not None:
        candidate_lags = [int(l) for l in exog_lags] if exog_lags is not None else [0, 1]
        unique_lags = []
        for lag in candidate_lags:
            if lag not in unique_lags:
                unique_lags.append(lag)
        exog_candidates = []
        for lag in unique_lags:
            try:
                series_to_align = extra_exog.shift(int(lag)) if lag else extra_exog
                exog_arr = series_to_align.reindex(y.index).astype(float).to_numpy()
                exog_arr = np.where(np.isfinite(exog_arr), exog_arr, 0.0)
                exog_candidates.append((int(lag), exog_arr))
            except Exception:
                continue
        if not exog_candidates:
            exog_candidates = [(None, None)]

    # K grid to do grid search for carrying capacity
    max_s = float(input_series.max())
    if k_grid is None:
        # Ensure K is strictly greater than max_s with a relative epsilon to avoid degeneracy
        baseline = max_s if max_s > 0.0 else 1.0
        eps = max(baseline * 1e-3, 1e-6)
        start = baseline + eps
        k_grid = np.concatenate(
            [
                np.linspace(start, baseline * 1.5 + eps, 8),
                np.linspace(baseline * 1.5 + eps, baseline * 5.0 + eps, 25),
                np.linspace(baseline * 6.0 + eps, baseline * 10.0 + eps, 10),
            ]
        )

    best: PiecewiseLogisticFit | None = None
    best_sse = np.inf
    best_score = np.inf
    best_exog_lag: int | None = None

    # Precompute Δ-space segment masks for efficiency and stability
    masks: list[np.ndarray] = []
    for start, end in seg_bounds:
        mask = np.zeros(n, dtype=float)
        lo = max(0, start)
        hi = min(n, end + 1)
        if lo < hi:
            mask[lo:hi] = 1.0
        masks.append(mask)

    # Indicator masks for segment-specific intercept adjustments relative to the first segment
    intercept_masks: list[np.ndarray] = []
    if num_segments > 1:
        for mask in masks[1:]:
            intercept_masks.append(mask.copy())

    # Hoist loop-invariant computations
    s_lag_arr = s_lag.to_numpy().astype(float)
    y_vec = y.to_numpy().astype(float)
    ones_vec = np.ones(n, dtype=float)
    lam = 1e-6

    for exog_lag, exog in exog_candidates:
        # Columns count is constant across K for a given exogenous candidate
        base_col_count = len(masks) + 1 + len(intercept_masks) + 2 + (1 if exog is not None else 0)
        ridge_I = lam * np.eye(base_col_count)
        for K in k_grid:
            X_base = s_lag_arr * (1.0 - s_lag_arr / K)
            # Build design matrix from precomputed masks
            X_cols: list[np.ndarray] = [(X_base * m) for m in masks]
            X_cols.append(ones_vec)  # global intercept baseline
            for im in intercept_masks:
                X_cols.append(im)
            X_cols.append(pulse)
            X_cols.append(step)
            if exog is not None:
                X_cols.append(exog)
            X = np.column_stack(X_cols)

            # OLS with a tiny ridge for stability (helps when columns are nearly collinear)
            XtX = X.T @ X
            Xty = X.T @ y_vec
            try:
                beta = np.linalg.solve(XtX + ridge_I, Xty)
            except np.linalg.LinAlgError:
                # very rare; fall back to lstsq
                beta, _, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)

            # Unpack parameters
            r_segments = [float(b) for b in beta[:num_segments]]
            gamma_intercept = float(beta[num_segments])
            offset_count = max(num_segments - 1, 0)
            offsets: list[float] = [0.0]
            if offset_count:
                start_idx = num_segments + 1
                offsets.extend(float(beta[start_idx + i]) for i in range(offset_count))
            gamma_pulse_idx = num_segments + 1 + offset_count
            gamma_pulse = float(beta[gamma_pulse_idx])
            gamma_step = float(beta[gamma_pulse_idx + 1])
            beta_offset = gamma_pulse_idx + 2
            gamma_exog = float(beta[beta_offset]) if exog is not None and len(beta) > beta_offset else None

            segment_intercepts = [gamma_intercept + offsets[i] for i in range(num_segments)] if num_segments else []

            # Reconstruct fitted series by integrating predicted deltas from the linear model
            y_hat = X @ beta
            s_hat = np.empty(n + 1, dtype=float)
            s_hat[0] = float(input_series.iloc[0])
            for t in range(n):
                s_hat[t + 1] = max(s_hat[t] + y_hat[t], 0.0)
            fitted = pd.Series(s_hat, index=input_series.index)
            # Residuals on deltas (aligned to y index)
            resid = y - pd.Series(y_hat, index=y.index)
            sse = float(np.square(resid.to_numpy()).sum())
            tss = float(np.square(y.to_numpy() - float(y.mean())).sum())
            r2 = 1.0 - (sse / tss if tss > 0 else np.nan)

            # Soft penalty to discourage negative segment growth rates without forbidding them
            neg_components = np.clip(-np.asarray(r_segments, dtype=float), 0.0, None)
            if np.any(neg_components > 0):
                penalty_strength = 0.05  # weak prior favouring non-negative growth
                scale = tss if tss > 0 else float(np.square(y_vec).sum())
                neg_penalty = penalty_strength * scale * float(np.square(neg_components).sum())
            else:
                neg_penalty = 0.0
            score = sse + neg_penalty

            fit = PiecewiseLogisticFit(
                carrying_capacity=float(K),
                segment_growth_rates=r_segments,
                segment_intercepts=segment_intercepts,
                breakpoints=bps,
                gamma_pulse=gamma_pulse,
                gamma_step=gamma_step,
                fitted_series=fitted,
                residuals=resid,
                sse=sse,
                r2_on_deltas=float(r2),
                gamma_exog=gamma_exog,
                gamma_intercept=gamma_intercept,
                exog_lag=exog_lag,
            )

            # Select by penalized score, tie-break by raw SSE
            if (score < best_score) or (np.isclose(score, best_score) and sse < best_sse) or best is None:
                best_score = score
                best_sse = sse
                best = fit
                best_exog_lag = exog_lag

    if best is None:
        raise RuntimeError("Could not fit piecewise logistic model")

    # Ensure the reported best fit carries the winning lag when exogenous search succeeded
    if best_exog_lag is not None and getattr(best, "exog_lag", None) != best_exog_lag:
        best = PiecewiseLogisticFit(
            carrying_capacity=best.carrying_capacity,
            segment_growth_rates=best.segment_growth_rates,
            segment_intercepts=best.segment_intercepts,
            breakpoints=best.breakpoints,
            gamma_pulse=best.gamma_pulse,
            gamma_step=best.gamma_step,
            fitted_series=best.fitted_series,
            residuals=best.residuals,
            sse=best.sse,
            r2_on_deltas=best.r2_on_deltas,
            gamma_exog=best.gamma_exog,
            gamma_intercept=best.gamma_intercept,
            exog_lag=best_exog_lag,
        )

    return best


def forecast_piecewise_logistic(
    last_value: float,
    months_ahead: int,
    carrying_capacity: float,
    segment_growth_rate: float,
    segment_intercept: float = 0.0,
    gamma_pulse: float = 0.0,
    gamma_step_level: float = 0.0,
    gamma_exog: float | None = None,
    exog_future: Sequence[float] | None = None,
    pulse_sequence: Sequence[float] | None = None,
) -> np.ndarray:
    """Forward simulation for the last segment using the documented dynamics.

    Parameters mirror the fitted equation:

    ΔS_t = r * S_{t-1} (1 - S_{t-1} / K) + α + γ_pulse pulse_t + γ_step step_t + γ_exog x_t

    The `gamma_step_level` argument represents the current step regressor level
    (typically 0.0 or 1.0) multiplied by its coefficient and is assumed constant
    over the forecast horizon.  A `pulse_sequence` can be supplied to control
    future pulses; by default a one-time pulse is applied on the first step to
    match the historical behaviour.  When `exog_future` is provided and
    `gamma_exog` is not ``None``, the exogenous contribution is applied
    element-wise.
    """

    months = max(int(months_ahead), 0)
    if months == 0:
        return np.empty(0, dtype=float)

    values = [float(last_value)]

    if pulse_sequence is None:
        pulses = [1.0] + [0.0] * max(months - 1, 0)
    else:
        pulses = [float(p) for p in pulse_sequence[:months]]
        if len(pulses) < months:
            pulses.extend([0.0] * (months - len(pulses)))

    if exog_future is None:
        exog_vals = [0.0] * months
    else:
        exog_vals = [float(val) for val in exog_future[:months]]
        if len(exog_vals) < months:
            exog_vals.extend([0.0] * (months - len(exog_vals)))

    intercept = float(segment_intercept)
    step_level = float(gamma_step_level)
    growth_rate = float(segment_growth_rate)
    gamma_p = float(gamma_pulse)
    gamma_ex = None if gamma_exog is None else float(gamma_exog)
    capacity = float(carrying_capacity)

    for step_idx in range(months):
        prev = values[-1]
        if capacity <= 0:
            x_t = 0.0
        else:
            x_t = prev * (1.0 - prev / capacity)
        delta = growth_rate * x_t + intercept + step_level
        delta += gamma_p * pulses[step_idx]
        if gamma_ex is not None:
            delta += gamma_ex * exog_vals[step_idx]
        values.append(max(prev + delta, 0.0))

    return np.array(values[1:], dtype=float)


def fitted_series_from_params(
    total_series: pd.Series,
    breakpoints: list[int],
    carrying_capacity: float,
    segment_growth_rates: Sequence[float],
    events_df: pd.DataFrame | None = None,
    extra_exog: pd.Series | None = None,
    extra_exog_lag: int | None = None,
    gamma_pulse: float = 0.0,
    gamma_step: float = 0.0,
    gamma_exog: float | None = None,
    gamma_intercept: float | None = None,
    segment_intercepts: Sequence[float] | None = None,
) -> pd.Series:
    """
    This takes the parameters and uses uses them to predict the future.

    Applies the same discrete dynamic used in fitting, aligned to month-end index.
    """
    s = ensure_month_end_index(total_series)
    if s.size == 0:
        return s

    # Align helper arrays to deltas index
    y_index = s.index[1:]
    pulse, step = _event_regressors(y_index, events_df)

    exog = None
    if extra_exog is not None:
        try:
            series_to_align = extra_exog.shift(int(extra_exog_lag)) if extra_exog_lag else extra_exog
            exog = series_to_align.reindex(y_index).astype(float).to_numpy()
            exog = np.where(np.isfinite(exog), exog, 0.0)
        except Exception:
            exog = None

    # Sanitise breakpoints in the same manner as the fitter
    n_series = len(s)
    if breakpoints:
        bps = sorted({int(b) for b in breakpoints if 1 <= int(b) <= max(n_series - 2, 1)})
    else:
        bps = []

    # Segment bounds on the original series index
    seg_bounds = _segments_from_breakpoints(n_series, bps)
    r_list = list(segment_growth_rates)
    if len(r_list) < len(seg_bounds):
        # Pad with last known rate
        r_list = r_list + [r_list[-1] if r_list else 0.0] * (len(seg_bounds) - len(r_list))

    num_segments = len(seg_bounds)

    # Determine per-segment intercepts
    if segment_intercepts is not None and len(segment_intercepts) > 0:
        intercepts = [float(v) for v in segment_intercepts]
        if len(intercepts) < num_segments:
            intercepts.extend([intercepts[-1]] * (num_segments - len(intercepts)))
        gamma_intercept = float(intercepts[0])
    else:
        if gamma_intercept is None:
            y = s.diff().dropna()
            s_lag = s.shift(1).reindex(y.index).astype(float)
            # Map each delta row to its segment index
            seg_idx_per_row: list[int] = [
                next((j for j, (a, b) in enumerate(seg_bounds) if a <= t <= b), 0) for t in range(y.size)
            ]

            x_base = s_lag.to_numpy() * (1.0 - s_lag.to_numpy() / carrying_capacity)
            contrib = np.zeros_like(x_base, dtype=float)
            for t, seg_idx in enumerate(seg_idx_per_row):
                contrib[t] = float(r_list[seg_idx]) * x_base[t]

            contrib += gamma_pulse * pulse
            contrib += gamma_step * step
            if exog is not None and gamma_exog is not None:
                contrib += gamma_exog * exog

            residual = y.to_numpy(dtype=float) - contrib
            gamma_intercept = float(np.nanmean(residual)) if residual.size else 0.0
        else:
            gamma_intercept = float(gamma_intercept)
        intercepts = [gamma_intercept] * max(num_segments, 1)

    intercepts = intercepts if num_segments else []

    # Build the same design matrix used in fitting to compute predicted deltas
    y = s.diff().dropna()
    s_lag = s.shift(1).reindex(y.index).astype(float)
    n = len(y)
    X_base = (s_lag * (1.0 - s_lag / carrying_capacity)).to_numpy(dtype=float)

    masks: list[np.ndarray] = []
    for start, end in seg_bounds:
        mask = np.zeros(n, dtype=float)
        lo = max(0, start)
        hi = min(n, end + 1)
        if lo < hi:
            mask[lo:hi] = 1.0
        masks.append(mask)
    intercept_masks = [m.copy() for m in masks[1:]] if len(masks) > 1 else []

    X_cols: list[np.ndarray] = [(X_base * m) for m in masks]
    X_cols.append(np.ones(n, dtype=float))
    for im in intercept_masks:
        X_cols.append(im)
    pulse_arr = np.asarray(pulse, dtype=float)
    step_arr = np.asarray(step, dtype=float)
    X_cols.append(pulse_arr)
    X_cols.append(step_arr)
    if exog is not None:
        X_cols.append(exog)

    if not X_cols:
        return s.astype(float)

    X = np.column_stack(X_cols)
    beta = np.zeros(X.shape[1], dtype=float)
    beta[:num_segments] = [float(r) for r in r_list[:num_segments]]
    if num_segments:
        baseline_intercept = intercepts[0]
    else:
        baseline_intercept = float(gamma_intercept or 0.0)
    beta[num_segments] = baseline_intercept
    for idx, intercept in enumerate(intercepts[1:], start=1):
        beta[num_segments + idx] = float(intercept - baseline_intercept)
    offset_count = max(num_segments - 1, 0)
    gamma_pulse_idx = num_segments + 1 + offset_count
    beta[gamma_pulse_idx] = gamma_pulse
    beta[gamma_pulse_idx + 1] = gamma_step
    if exog is not None:
        beta[gamma_pulse_idx + 2] = gamma_exog if gamma_exog is not None else 0.0

    y_hat = X @ beta
    s_hat = np.empty(n + 1, dtype=float)
    s_hat[0] = float(s.iloc[0])
    for t in range(n):
        s_hat[t + 1] = max(s_hat[t] + y_hat[t], 0.0)
    return pd.Series(s_hat, index=s.index)
