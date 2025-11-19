## Headless output bundle

Running `python scripts/run_headless.py ... --out-dir ./outputs` produces the same Phase 1 artifacts that the Streamlit app saves. Key files:

- `summary.json` – detector configuration, chosen breakpoints, fit parameters (`carrying_capacity`, `segment_growth_rates`, `segment_intercepts`, `gamma_*`), and fit diagnostics (`sse`, `r2_on_deltas`). It also includes the derived adds/churn estimates used for downstream stages.
- `fitted_series.csv` – deterministic piecewise-logistic fit at the supplied cadence with a `fitted` column.
- `features.csv` and `covariates.csv` – engineered regressors: adstocked/log ad response plus pulse/step encodings from events.
- `events_normalized.csv` – cleaned event table with normalized types, persistence, and optional cost; only emitted when an events CSV was supplied.
- `equation.md` – the rendered growth equation with the fitted parameters and the inputs that fed the model.
- `phase1.json` – portable artifact that you can load into the app (Stage 2) to continue with simulation without re-running the fit.


