# Piecewise logistic fit (Phase 1)

This document is the authoritative description of the model that the calibration
pipeline fits today.  The goal is to recover a segmented logistic growth curve on
monthly **total subscribers** while allowing for both transient and persistent
externally supplied events and an optional exogenous driver.  The fitter is
fully deterministic once the inputs are supplied; there is no stochastic noise
term.

## Core dynamics
For each month-end observation we model the change in total subscribers as

\[
\Delta S_t = \sum_{j=1}^{J} r_j \mathbf{1}_{t \in \mathcal{S}_j} \cdot X_t(K)
             + \sum_{j=1}^{J} \alpha_j \mathbf{1}_{t \in \mathcal{S}_j}
             + \gamma_{\text{pulse}} P_t
             + \gamma_{\text{step}} L_t
             + \gamma_{\text{exog}} E_t,
\]

where

- \(X_t(K) = S_{t-1} \bigl(1 - S_{t-1} / K\bigr)\) is the logistic regressor
  evaluated at the **global** carrying capacity \(K\) selected via grid-search.
- The index set \(\mathcal{S}_j\) contains the delta rows belonging to segment
  \(j\).  Segments are constructed from the supplied change points (or from the
  detector) and allow each growth rate \(r_j\) to shift piecewise over time.
- \(\alpha_j\) is the segment intercept.  The fitter implements this as a
  global intercept plus per-segment offsets so that level drift can differ
  across regimes even when the logistic term is near zero.
- \(P_t\) is the **pulse** regressor: a one-month spike (1 in the event month,
  0 otherwise).
- \(L_t\) is the **step** regressor: it turns on in the event month and remains
  at 1 thereafter to capture persistent shifts.
- \(E_t\) is an optional exogenous series supplied by the caller.  When an
  extra feature is provided the fitter evaluates a small set of integer lags and
  selects the best-performing alignment; otherwise this term is omitted.

If no exogenous series is provided we simply drop the
\(\gamma_{\text{exog}} E_t\) term.  Likewise, if the event tables are empty the
pulse and step terms are identically zero.

## What the fitter returns
Running `fit_piecewise_logistic(...)` produces:

- the selected carrying capacity \(K\);
- one growth rate \(r_j\) and intercept \(\alpha_j\) for each segment;
- the pulse and step coefficients \(\gamma_{\text{pulse}}\) and
  \(\gamma_{\text{step}}\);
- the optional exogenous coefficient and lag (when a feature was provided);
- the deterministic fitted series obtained by integrating the predicted
  \(\Delta S_t\) values.

These outputs are packaged in `PiecewiseLogisticFit` and surfaced in the UI and
handoff artifacts.

## Notes and limitations

- The regression is solved with ordinary least squares plus a tiny ridge term
  for numerical stability; no random noise component is estimated.
- Carrying capacity is shared across segments, but growth rates and intercepts
  can change whenever the change-point detector inserts a new segment.
- Event handling treats ads or campaigns entered as **events** as pulses or
  steps depending on the chosen persistence.  Continuous spend should instead be
  passed through the exogenous channel if you want it modeled via
  \(\gamma_{\text{exog}}\).
- This fitter operates purely on total subscriber counts.  The simulator uses a
  richer free/premium cohort decomposition; see the simulator documentation for
  those dynamics.

By grounding the documentation in the model that actually ships, we avoid the
confusion caused by the older conceptual equation that omitted intercepts,
change points, and step events.
