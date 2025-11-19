# Substack Analyzer

## How to use

Open the hosted app: https://substackanalyzer.streamlit.app/ (just click to wake it up if it's gone to sleep). The Streamlit UI walks through Phase 1 (fit and diagnostics) and Phase 2 (simulation and outputs) with inline downloads and a Save / Load tab for session bundles.

### Headless (Python-only) runner

You can run the Phase 1 pipeline from CSV/XLSX files without the UI:

```bash
python scripts/run_headless.py \
  --all /path/to/total.csv --all-has-header --all-date-col 0 --all-count-col 1 \
  --paid /path/to/paid.csv --paid-has-header --paid-date-col 0 --paid-count-col 1 \
  --events /path/to/events.csv \
  --adspend /path/to/ad_spend.csv \
  --max-changes 4 --detect-on auto --detector classifier \
  --lam 0.5 --theta 500 \
  --out-dir ./outputs
```

Notes:

- Provide at least one of `--all` or `--paid`. Each should point to a two-column file (date, count). Use `--*-has-header` and `--*-date-col`/`--*-count-col` to specify header presence and column indices or names.
- `--events` (optional) CSV columns: `date`, `type`, `persistence`, `cost`. Events are encoded as pulse/step regressors for the logistic fit and saved to `events_normalized.csv` if present.
- `--adspend` (optional) CSV/XLSX columns: `date`, `spend`. The ad stock/log-response parameters (`--lam`, `--theta`) control the exogenous regressor used in the fit.
- Change-point detection matches the Streamlit app: `--detector classifier` (default) or `--detector simple`, `--detect-on` for total/paid/free/both, plus `--min-seg-len`, `--penalty-scale`, `--window`, and `--z-pulse` tuning knobs. Add `--keep-all-breaks` to skip classifier-based filtering of persistent breakpoints.
- Outputs in `--out-dir` include:
  - `summary.json`: detected breakpoints, fit parameters, fit diagnostics, and derived estimates.
  - `fitted_series.csv`: deterministic piecewise-logistic fit on the supplied cadence.
  - `covariates.csv` / `features.csv`: engineered regressors used in the fit.
  - `events_normalized.csv`: cleaned events used for encoding (when provided).
  - `equation.md`: human-readable growth equation with parameters and inputs.
  - `phase1.json`: the portable artifact you can load into the app to proceed to the simulator.

## What this does

- Simulates monthly subscriber growth for free and premium cohorts
- Combines organic growth, paid acquisition (ad spend / CAC with diminishing returns), and churn
- Converts a share of new free subs to premium immediately, plus a small ongoing conversion of existing free
- Computes net revenue after Substack (10%) and Stripe (3.6% + $0.30) fees
- Tracks profit = net revenue − ad spend − ad manager fee
- Visualizes KPIs, subscribers over time, spend vs revenue

## Mapping Substack stats to inputs

- Organic monthly growth (free):
  - Use Substack's "Subscribers over time" or exports to estimate average monthly net new free subs excluding paid acquisition. Divide by the free base to get a rate.
- Cost per new free subscriber (CAC):
  - From your ad platform data or Substack "Where subscribers came from" exports when tagged. CAC = ad spend / new free subs attributed to ads.
- New-subscriber premium conversion:
  - Estimate the share of newly acquired free subs who convert to premium within the first month. If unknown, start with 1–3%.
- Ongoing premium conversion of existing free:
  - Small monthly rate applied to the existing free base. If unknown, start with 0.02–0.05% (0.0002–0.0005).
- Churn (free and premium):
  - Use list cleaning + unsubscribes divided by cohort size monthly. If you only have paid churn, apply that to premium and set free churn around 0.5–2%.
- Pricing and fees:
  - Substack fee 10%, Stripe 3.6% + $0.30 are defaults; update as needed.
- Ad spend schedule:
  - Two-stage lets you specify a budget for year 1 and a different budget for year 2 and beyond; constant uses a flat monthly spend.

## Notes and limitations (MVP)

- Conversions and churn are applied at a monthly granularity with simplified timing.
- Annual plan revenue is amortized evenly across months if enabled.
- Attribution/organic separation will vary by how you tag campaigns and sources.


See the design document for the end-to-end flow, stage inputs/outputs, and data contracts: [DESIGN.md](./DESIGN.md)
