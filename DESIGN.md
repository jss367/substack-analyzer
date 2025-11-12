# Substack Growth Modeling — System Design

[IN PROGRESS]

This document captures the shipping scope of the Substack growth analysis tool. The
current product centers on a deterministic pipeline: clean the Substack exports,
annotate events, derive heuristic adds/churn estimates, fit a piecewise logistic
model, and feed those results into a cohort simulator with export tooling. Longer-
term ideas still exist, but anything not listed here should be treated as out of
scope.

---

## What’s Included Today

- **Adds vs. churn separation** derived from totals using heuristic rates.
- **Piecewise logistic fit** with change-points sourced from the detection module.
- **Events and adstock features** that flow consistently from import → fit → simulator.
- **Deterministic cohort simulator** with finance metrics and export bundles.
- **Manual and automated exports** for handoffs between phases.

---

## Phase 1 — Fit & Diagnostics

### Stage 1 — File Upload & Parsing

**Inputs**

- Substack _All Subscribers_ export (CSV/XLSX).
- Substack _Paid Subscribers_ export (CSV/XLSX).
- Optional starter bundles (`phase1.json`) captured by earlier sessions.

**Processing**

- Load the files with flexible column selection.
- Normalize to month-end cadence with explicit imputation flags.
- Derive Free = Total − Paid, clipped at zero.

**Outputs**

- `observations_df`: monthly totals (paid, free, total) + metadata.
- Download link for the cleaned table.

---

### Stage 2 — Events & Features

**Inputs**

- Manual event table captured in the UI (`events_df`).
- Suggested change-points from the detection helper.
- Sidebar inputs for adstock λ and log-response θ.

**Processing**

- Normalize event types and persistence (Transient, Persistent, None).
- Build adstocked spend series and log-response features.
- Emit combined feature sets for later modules (`covariates_df`, `features_df`).

**Outputs**

- Editable events grid with import/export of `phase1.json`.
- Downloadable CSVs of engineered features.

---

### Stage 3 — Adds & Churn

**Processing**

- Apply heuristic churn estimates supplied in the sidebar.
- Split net deltas into `adds_df` and `churn_df` tables.
- Provide CSV downloads for validation outside the app.

---

### Stage 4 — Quick Fit

**Processing**

- Run change-point detection to segment the piecewise logistic.
- Fit capacity, growth, and event coefficients with optional ad-response term.
- Surface override sliders for manual adjustments.

**Outputs**

- Fitted parameters, overlay charts, and forward projections.
- Growth equation text block for downstream reference.

---

### Stage 5 — Diagnostics

**Processing**

- Show delta charts with slope overlays and trailing-window metrics.
- Highlight quick estimators (growth rates, implied churn, saturation proximity).

**Outputs**

- Status recap for Stage 1–4 artifacts.
- Download of the assembled `phase1.json` bundle.

_Future work:_ richer cross-validation beyond the current summary checks.

---

## Phase 2 — Simulation & Outputs

### Stage 6 — Cohort & Finance Simulator

**Inputs**

- `phase1.json` or a full session bundle from Phase 1.
- Scenario levers: growth modifiers, churn overrides, ad spend schedules, conversion settings.

**Processing**

- Run deterministic free/paid cohort evolution with tenure-based churn curves.
- Model free→paid conversion and ad-driven acquisition using Phase 1 estimates.
- Compute finance KPIs (MRR/ARR, ROAS, CAC, payback).

**Outputs**

- Scenario tables and charts rendered inside Streamlit.
- KPI tiles summarizing cohort health and financial outcomes.

---

### Stage 7 — Outputs & Documentation

**Processing**

- Export `phase1.json`, simulator CSVs, and zipped session bundles.
- Surface equation snippets and definitions for the deterministic pipeline.

**Outputs**

- Downloadable artifacts for further analysis or reporting.
- Lightweight reference documentation embedded in the UI.

_Future work:_ richer automated documentation built from session artifacts.

---

## Data Contracts (Current)

- `observations_df`: `date`, `active_total`, `active_paid`, `active_free`, `is_imputed`.
- `events_df`: `date`, `type`, `persistence`, `notes`, `cost`.
- `features_df`: `date`, `adstock`, `ad_effect`, `pulse`, plus derived covariates.
- `adds_df` / `churn_df`: monthly adds and cancels derived from totals.
- `simulator_inputs`: serialized configuration for the cohort model.

---

## Out of Scope

Any advanced probabilistic modeling workstreams are tracked separately and should
not be assumed to exist in the deterministic application described here.
