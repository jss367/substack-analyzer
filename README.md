# Substack Analyzer

## How to use

Open the hosted app: https://substackanalyzer.streamlit.app/ (just click to wake it up if it's gone to sleep)

### Headless (Python-only) runner

You can run the core pipeline from CSV/XLSX files without the UI:

```bash
python scripts/run_headless.py \
  --all /path/to/total.csv --all-has-header --all-date-col 0 --all-count-col 1 \
  --paid /path/to/paid.csv --paid-has-header --paid-date-col 0 --paid-count-col 1 \
  --events /path/to/events.csv \
  --adspend /path/to/ad_spend.csv \
  --max-changes 4 --lam 0.5 --theta 500 \
  --out-dir ./outputs
```

Notes:

- `--all` and/or `--paid` should each point to a two-column file (date, count). Use `--*-has-header` and `--*-date-col`/`--*-count-col` to specify header presence and column indices or names.
- `--events` (optional) CSV columns: `date`, `type`, `persistence`, `cost`.
- `--adspend` (optional) CSV/XLSX columns: `date`, `spend`.
- Change-point detector settings (classifier vs. simple, filtering, sensitivity) are documented in [docs/detection-modes.md](docs/detection-modes.md).
- Outputs include `summary.json`, `fitted_series.csv`, `features.csv`, and `covariates.csv` in `--out-dir`.

## What this does

- Simulates monthly subscriber growth for free and premium cohorts
- Combines organic growth, paid acquisition (ad spend / CAC), and churn
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
  - Two-stage lets you specify a higher budget in years 1–2 and lower in years 3–5; constant uses a flat monthly spend.

## Notes and limitations (MVP)

- Conversions and churn are applied at a monthly granularity with simplified timing.
- Annual plan revenue is amortized evenly across months if enabled.
- Attribution/organic separation will vary by how you tag campaigns and sources.


See the design document for the end-to-end flow, stage inputs/outputs, and data contracts: [DESIGN.md](./DESIGN.md)
