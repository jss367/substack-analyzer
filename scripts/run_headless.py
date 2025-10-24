"""
Headless runner for Substack Analyzer.

Usage examples:
  python scripts/run_headless.py \
    --all /path/to/total.csv --all-has-header --all-date-col 0 --all-count-col 1 \
    --paid /path/to/paid.csv --paid-has-header --paid-date-col 0 --paid-count-col 1 \
    --events /path/to/events.csv \
    --adspend /path/to/ad_spend.csv \
    --max-changes 4 --lam 0.5 --theta 500 \
    --out-dir ./outputs

The All/Paid CSV/XLSX should have date and count columns. You can specify columns
by index (0-based) or by name when headers are present.
Events CSV is optional and should have columns at least: date, type, persistence, cost.
Ad spend CSV/XLSX is optional with columns: date, spend.
"""



import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from substack_analyzer.analysis import build_events_features, compute_estimates, read_series
from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.detection import detect_change_points


def _open_file(path: Optional[str]) -> Optional[object]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.open("rb")


def _read_events_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Accept flexible column sets; normalise column names
    cols = {c.lower().strip(): c for c in df.columns}

    def get(col: str) -> Optional[str]:
        return cols.get(col)

    out = pd.DataFrame()
    if get("date") is None:
        raise ValueError("Events file must include a 'date' column")
    out["date"] = pd.to_datetime(df[get("date")], errors="coerce")
    out["type"] = df[get("type")] if get("type") in df else "Event"
    if get("persistence") in df:
        out["persistence"] = df[get("persistence")]
    if get("cost") in df:
        out["cost"] = pd.to_numeric(df[get("cost")], errors="coerce")
    out = out.dropna(subset=["date"])  # require valid date
    return out


def _ensure_month_end_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    s = s.sort_index()
    return s


def run(
    all_path: Optional[str],
    all_has_header: bool,
    all_date_col,
    all_count_col,
    paid_path: Optional[str],
    paid_has_header: bool,
    paid_date_col,
    paid_count_col,
    events_path: Optional[str],
    adspend_path: Optional[str],
    max_changes: int,
    lam: float,
    theta: float,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Read series
    total: Optional[pd.Series] = None
    paid: Optional[pd.Series] = None
    if all_path:
        with _open_file(all_path) as f_all:
            total = read_series(f_all, all_has_header, all_date_col, all_count_col)
            total = _ensure_month_end_index(total)
    if paid_path:
        with _open_file(paid_path) as f_paid:
            paid = read_series(f_paid, paid_has_header, paid_date_col, paid_count_col)
            paid = _ensure_month_end_index(paid)

    if total is None and paid is None:
        raise SystemExit("Provide at least one of --all or --paid")

    # Seed minimal session state for downstream code
    st.session_state.clear()
    st.session_state["start_premium"] = int(paid.iloc[-1]) if isinstance(paid, pd.Series) and not paid.empty else 0

    # Optional events
    events_df = _read_events_csv(events_path) if events_path else pd.DataFrame()
    st.session_state["events_df"] = events_df

    # Timeline index from whichever series is present
    if total is not None and not total.empty:
        idx = total.index
    elif paid is not None and not paid.empty:
        idx = paid.index
    else:
        raise SystemExit("No data in provided series")

    plot_df = pd.DataFrame(index=idx)
    if total is not None:
        plot_df["Total"] = total.reindex(idx)
    if paid is not None:
        plot_df["Paid"] = paid.reindex(idx)

    # Detection runs on Total if available, otherwise on Paid
    target_for_detection = plot_df["Total"] if "Total" in plot_df.columns else plot_df["Paid"]
    bkps = detect_change_points(target_for_detection, max_changes=max_changes, min_seg_len=3, return_mode="indices")

    # Features and optional ad spend
    ad_file_handle = _open_file(adspend_path) if adspend_path else None
    covariates_df, features_df = build_events_features(plot_df, lam=lam, theta=theta, ad_file=ad_file_handle)
    exog = features_df["ad_effect_log"] if "ad_effect_log" in features_df else None

    # Estimates
    est = compute_estimates(all_series=plot_df.get("Total"), paid_series=plot_df.get("Paid"), window_months=6)

    # Fit model on Total if present, else Paid
    fit_series = plot_df["Total"] if "Total" in plot_df.columns else plot_df["Paid"]
    fit = fit_piecewise_logistic(
        total_series=fit_series,
        breakpoints=bkps,
        events_df=st.session_state.get("events_df"),
        extra_exog=exog,
    )

    # Save outputs
    out_summary = {
        "breakpoints": bkps,
        "carrying_capacity": fit.carrying_capacity,
        "segment_growth_rates": fit.segment_growth_rates,
        "gamma_pulse": fit.gamma_pulse,
        "gamma_step": fit.gamma_step,
        "gamma_exog": fit.gamma_exog,
        "gamma_intercept": fit.gamma_intercept,
        "sse": fit.sse,
        "r2_on_deltas": fit.r2_on_deltas,
        "estimates": est,
    }
    (Path(out_dir) / "summary.json").write_text(json.dumps(out_summary, indent=2))

    fit.fitted_series.to_csv(Path(out_dir) / "fitted_series.csv", header=["fitted"], index_label="date")
    if not events_df.empty:
        events_df.to_csv(Path(out_dir) / "events_normalized.csv", index=False)
    covariates_df.to_csv(Path(out_dir) / "covariates.csv", index_label="date")
    features_df.to_csv(Path(out_dir) / "features.csv", index_label="date")

    # Print short human-readable summary
    print("=== Fit summary ===")
    print(json.dumps(out_summary, indent=2))
    print(f"Outputs written to: {Path(out_dir).resolve()}")


def _col_arg(s: str) -> str | int:
    # Helper: parse column arg as int if possible, else keep as string (name)
    try:
        return int(s)
    except (ValueError, TypeError):
        return s


def main() -> None:
    p = argparse.ArgumentParser(description="Headless runner for Substack Analyzer")
    p.add_argument("--all", dest="all_path", type=str, help="Path to All/Total CSV/XLSX", default=None)
    p.add_argument("--all-has-header", action="store_true", help="All file has header row")
    p.add_argument("--all-date-col", type=_col_arg, default=0, help="All date column (index or name)")
    p.add_argument("--all-count-col", type=_col_arg, default=1, help="All count column (index or name)")

    p.add_argument("--paid", dest="paid_path", type=str, help="Path to Paid CSV/XLSX", default=None)
    p.add_argument("--paid-has-header", action="store_true", help="Paid file has header row")
    p.add_argument("--paid-date-col", type=_col_arg, default=0, help="Paid date column (index or name)")
    p.add_argument("--paid-count-col", type=_col_arg, default=1, help="Paid count column (index or name)")

    p.add_argument("--events", dest="events_path", type=str, default=None, help="Path to Events CSV")
    p.add_argument("--adspend", dest="adspend_path", type=str, default=None, help="Path to Ad spend CSV/XLSX")

    p.add_argument("--max-changes", type=int, default=4, help="Max change points to detect")
    p.add_argument("--lam", type=float, default=0.5, help="Adstock lambda carryover [0,1)")
    p.add_argument("--theta", type=float, default=500.0, help="Ad effect log scale")
    p.add_argument("--out-dir", type=str, default="./outputs", help="Output directory")

    args = p.parse_args()

    run(
        all_path=args.all_path,
        all_has_header=args.all_has_header,
        all_date_col=args.all_date_col,
        all_count_col=args.all_count_col,
        paid_path=args.paid_path,
        paid_has_header=args.paid_has_header,
        paid_date_col=args.paid_date_col,
        paid_count_col=args.paid_count_col,
        events_path=args.events_path,
        adspend_path=args.adspend_path,
        max_changes=args.max_changes,
        lam=args.lam,
        theta=args.theta,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
