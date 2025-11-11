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
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import coloredlogs
import pandas as pd
import streamlit as st

from substack_analyzer.analysis import (
    DEFAULT_AD_LOG_THETA,
    DEFAULT_ADSTOCK_LAMBDA,
    build_events_features,
    compute_estimates,
    read_series,
)
from substack_analyzer.calibration import fit_piecewise_logistic
from substack_analyzer.detection import detect_change_points
from substack_analyzer.persistence import export_phase_one_json
from substack_analyzer.utils import coerce_list, ensure_month_end_index

coloredlogs.install(level="DEBUG")
logger = logging.getLogger("substack_headless")


@dataclass(frozen=True)
class PhaseOneConfig:
    all_path: str | None
    all_has_header: bool
    all_date_col: str | int
    all_count_col: str | int
    paid_path: str | None
    paid_has_header: bool
    paid_date_col: str | int
    paid_count_col: str | int
    events_path: str | None
    adspend_path: str | None
    max_changes: int
    detect_on: str
    lam: float
    theta: float
    out_dir: str


def run_from_phase_one_config(cfg: PhaseOneConfig) -> None:
    run(
        all_path=cfg.all_path,
        all_has_header=cfg.all_has_header,
        all_date_col=cfg.all_date_col,
        all_count_col=cfg.all_count_col,
        paid_path=cfg.paid_path,
        paid_has_header=cfg.paid_has_header,
        paid_date_col=cfg.paid_date_col,
        paid_count_col=cfg.paid_count_col,
        events_path=cfg.events_path,
        adspend_path=cfg.adspend_path,
        max_changes=cfg.max_changes,
        detect_on=cfg.detect_on,
        lam=cfg.lam,
        theta=cfg.theta,
        out_dir=cfg.out_dir,
    )


def _open_file(path: str | None) -> object | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.open("rb")


def _read_events_csv(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Accept flexible column sets; normalise column names
    cols = {c.lower().strip(): c for c in df.columns}

    def get(col: str) -> str | None:
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


def run(
    all_path: str | None,
    all_has_header: bool,
    all_date_col,
    all_count_col,
    paid_path: str | None,
    paid_has_header: bool,
    paid_date_col,
    paid_count_col,
    events_path: str | None,
    adspend_path: str | None,
    max_changes: int,
    detect_on: str,
    lam: float,
    theta: float,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Starting headless run")
    logger.debug(
        "Args: all=%s, paid=%s, events=%s, adspend=%s, detect_on=%s, lam=%.3f, theta=%.3f, out_dir=%s",
        all_path,
        paid_path,
        events_path,
        adspend_path,
        detect_on,
        lam,
        theta,
        out_dir,
    )

    # Read series
    total: pd.Series | None = None
    paid: pd.Series | None = None
    if all_path:
        with _open_file(all_path) as f_all:
            total = read_series(f_all, all_has_header, all_date_col, all_count_col)
            total = ensure_month_end_index(total)
        if total is not None and not total.empty:
            logger.info(
                "Loaded ALL series: %d rows, %s → %s",
                len(total),
                str(total.index.min().date()),
                str(total.index.max().date()),
            )
    if paid_path:
        with _open_file(paid_path) as f_paid:
            paid = read_series(f_paid, paid_has_header, paid_date_col, paid_count_col)
            paid = ensure_month_end_index(paid)
        if paid is not None and not paid.empty:
            logger.info(
                "Loaded PAID series: %d rows, %s → %s",
                len(paid),
                str(paid.index.min().date()),
                str(paid.index.max().date()),
            )

    if total is None and paid is None:
        raise SystemExit("Provide at least one of --all or --paid")

    # Seed minimal session state for downstream code
    st.session_state.clear()
    st.session_state["start_premium"] = int(paid.iloc[-1]) if isinstance(paid, pd.Series) and not paid.empty else 0
    # Persist series so Phase 1 export includes them
    st.session_state["import_total"] = total if isinstance(total, pd.Series) else pd.Series(dtype=float)
    st.session_state["import_paid"] = paid if isinstance(paid, pd.Series) else pd.Series(dtype=float)

    # Optional events
    events_df = _read_events_csv(events_path) if events_path else pd.DataFrame()
    st.session_state["events_df"] = events_df
    if events_path:
        logger.info("Loaded EVENTS from %s: %d rows", events_path, 0 if events_df is None else len(events_df))

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

    # Optionally compute Free if we have both Total and Paid
    if {"Total", "Paid"}.issubset(plot_df.columns):
        plot_df["Free"] = (plot_df["Total"].astype(float) - plot_df["Paid"].astype(float)).clip(lower=0)
    logger.info("Built plot_df with columns: %s (rows=%d)", ", ".join(plot_df.columns), len(plot_df.index))

    # Detection target selection and optional merge

    def _detect_indices(series: pd.Series) -> list[int]:
        return detect_change_points(series.dropna(), max_changes=max_changes, min_seg_len=3, return_mode="indices")

    def _indices_to_dates(series: pd.Series, indices: list[int]) -> list[pd.Timestamp]:
        s_idx = series.dropna().index
        return [s_idx[i] for i in indices if 0 <= i < len(s_idx)]

    def _merge_dates(dates: list[pd.Timestamp], min_gap_months: int = 1) -> list[pd.Timestamp]:
        if not dates:
            return []
        ds = sorted({pd.to_datetime(d).to_period("M").to_timestamp("M") for d in dates})
        merged: list[pd.Timestamp] = []
        for d in ds:
            if not merged:
                merged.append(d)
            else:
                prev = merged[-1]
                # enforce minimum gap in months
                if (d.to_period("M").ordinal - prev.to_period("M").ordinal) >= min_gap_months:
                    merged.append(d)
        return merged

    # Choose detection according to mode
    bkps: list[int] = []
    detect_mode = detect_on.lower()
    # Default base for mapping indices in fit
    fit_series = plot_df["Total"] if "Total" in plot_df.columns else plot_df.get("Paid", plot_df.iloc[:, 0])

    if detect_mode in {"auto", "default"}:
        target = (
            plot_df["Total"]
            if "Total" in plot_df.columns
            else (plot_df["Paid"] if "Paid" in plot_df.columns else plot_df.get("Free"))
        )
        if target is None:
            raise SystemExit("No suitable series for detection (Auto mode)")
        bkps = _detect_indices(target)
    elif detect_mode == "total":
        if "Total" not in plot_df.columns:
            raise SystemExit("Total series not available for detection")
        bkps = _detect_indices(plot_df["Total"])
        fit_series = plot_df["Total"]
    elif detect_mode == "paid":
        if "Paid" not in plot_df.columns:
            raise SystemExit("Paid series not available for detection")
        bkps = _detect_indices(plot_df["Paid"])
        fit_series = plot_df["Paid"]
    elif detect_mode == "free":
        if "Free" not in plot_df.columns:
            raise SystemExit("Free series not available for detection (need Total and Paid)")
        bkps = _detect_indices(plot_df["Free"])
        # Fit preference stays: Total if present else Paid else Free
    elif detect_mode == "both":
        if not {"Total", "Paid"}.issubset(plot_df.columns):
            raise SystemExit("Both mode requires Total and Paid series")
        b_total = _detect_indices(plot_df["Total"])
        b_paid = _detect_indices(plot_df["Paid"])
        d_total = _indices_to_dates(plot_df["Total"], b_total)
        d_paid = _indices_to_dates(plot_df["Paid"], b_paid)
        d_merged = _merge_dates(d_total + d_paid, min_gap_months=1)
        base_index = plot_df["Total"].dropna().index if "Total" in plot_df.columns else plot_df["Paid"].dropna().index
        bkps = [base_index.get_loc(d) for d in d_merged if d in base_index]
        # Keep bkps sorted and unique
        bkps = sorted(set(bkps))
    else:
        raise SystemExit(f"Unknown detect-on mode: {detect_mode}")
    logger.info("Detection mode=%s, breakpoints=%s", detect_mode, bkps)

    # Features and optional ad spend (use configured lam/theta)
    ad_file_handle = _open_file(adspend_path) if adspend_path else None
    lam_best = float(lam) if lam is not None else DEFAULT_ADSTOCK_LAMBDA
    theta_best = float(theta) if theta is not None else DEFAULT_AD_LOG_THETA
    covariates_df, features_df = build_events_features(
        plot_df, lam=lam_best, theta=theta_best, ad_file=ad_file_handle
    )

    if adspend_path:
        ad_sum = float(covariates_df["ad_spend"].sum()) if "ad_spend" in covariates_df.columns else 0.0
        logger.info(
            "Ad spend file loaded: %s (monthly rows=%d, total spend=%.2f)",
            adspend_path,
            len(covariates_df),
            ad_sum,
        )
    exog = features_df["ad_effect_log"] if "ad_effect_log" in features_df else None
    # Persist Phase 1 state for export
    st.session_state["covariates_df"] = covariates_df
    st.session_state["features_df"] = features_df
    st.session_state["adstock_lambda"] = float(lam_best)
    st.session_state["ad_log_theta"] = float(theta_best)
    st.session_state["detected_breakpoints"] = bkps
    # Map indices to dates based on chosen fit_series base
    try:
        s_idx = fit_series.dropna().index
        change_dates = [s_idx[i] for i in bkps if 0 <= i < len(s_idx)]
    except Exception:
        change_dates = []
    st.session_state["detected_change_dates"] = change_dates
    st.session_state["detect_on"] = detect_mode

    # Estimates
    est = compute_estimates(all_series=plot_df.get("Total"), paid_series=plot_df.get("Paid"), window_months=6)
    logger.info("Estimates: %s", json.dumps(est))

    # Fit model on Total if present, else Paid (or previously selected base)
    if "Total" in plot_df.columns:
        fit_series = plot_df["Total"]
    elif "Paid" in plot_df.columns:
        fit_series = plot_df["Paid"]
    fit = fit_piecewise_logistic(
        total_series=fit_series,
        breakpoints=bkps,
        events_df=st.session_state.get("events_df"),
        extra_exog=exog,
    )
    # Expose fit in session state so phase1.json export can include fit params
    st.session_state["pwlog_fit"] = fit
    try:
        k_disp = int(getattr(fit, "carrying_capacity", 0) or 0)
        r2 = float(getattr(fit, "r2_on_deltas", 0.0))
        logger.info("Fit done: K=%s, r2_on_deltas=%.3f", f"{k_disp:,}", r2)
    except Exception:
        logger.info("Fit done")

    # Save outputs
    out_summary = {
        "breakpoints": bkps,
        "detect_on": detect_mode,
        "breakpoints_total": (b_total if 'b_total' in locals() else None),
        "breakpoints_paid": (b_paid if 'b_paid' in locals() else None),
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
    out_dir_path = Path(out_dir)
    (out_dir_path / "summary.json").write_text(json.dumps(out_summary, indent=2))

    # Write artifacts
    try:
        fit.fitted_series.to_csv(out_dir_path / "fitted_series.csv", header=["fitted"], index_label="date")
    except Exception:
        logger.exception("Failed to write fitted_series.csv")
        raise
    ev_out = st.session_state.get("events_df")
    if isinstance(ev_out, pd.DataFrame) and not ev_out.empty:
        ev_out.to_csv(out_dir_path / "events_normalized.csv", index=False)
    covariates_df.to_csv(out_dir_path / "covariates.csv", index_label="date")
    features_df.to_csv(out_dir_path / "features.csv", index_label="date")

    # Save Phase 1 portable artifact
    try:
        p1_bytes = export_phase_one_json()
        (out_dir_path / "phase1.json").write_bytes(p1_bytes)
        logger.info("Phase 1 artifact saved: %s", str((out_dir_path / "phase1.json").resolve()))
        logger.info("You can load phase1.json in the app (Stage 2) to proceed to Phase 2 fit.")
    except Exception as e:
        logger.warning("Phase 1 artifact not saved: %s", e)

    # Human-readable equation document (markdown)
    try:
        eq = (
            r"$\\Delta S_t = r_{seg(t)} \\, S_{t-1} \\left(1 - \\frac{S_{t-1}}{K}\\right) "
            r"+ \\gamma_{pulse}\\,pulse_t + \\gamma_{step}\\,step_t$"
        )
        if getattr(fit, "gamma_exog", None) is not None:
            eq = eq[:-1] + r" + \\gamma_{exog}\\,x_t$"

        k_now = getattr(fit, "carrying_capacity", None)
        r_list = coerce_list(getattr(fit, "segment_growth_rates", None))
        gp = getattr(fit, "gamma_pulse", None)
        gs = getattr(fit, "gamma_step", None)
        gx = getattr(fit, "gamma_exog", None)

        lines: list[str] = []
        lines.append("# Growth equation (piecewise logistic)")
        lines.append("")
        lines.append(eq)
        lines.append("")
        lines.append("## Parameters")
        if k_now is not None:
            lines.append(f"- K (capacity): {float(k_now):,.0f}")
        if r_list:
            lines.append("- Segment growth rates r_j: " + ", ".join(f"{r:0.3f}" for r in r_list))
        if gp is not None:
            lines.append(f"- gamma_pulse: {float(gp):0.4f}")
        if gs is not None:
            lines.append(f"- gamma_step: {float(gs):0.4f}")
        if gx is not None:
            lines.append(f"- gamma_exog (log ad effect): {float(gx):0.4f}")
        lines.append(f"- Adstock lambda (lam): {float(lam_best):0.3f}")
        lines.append(f"- Log-scale theta: {float(theta_best):0.3f}")
        lines.append("")
        lines.append("## Inputs")
        lines.append("- x_t = features['ad_effect_log'] (built from ad_spend with adstock + log transform)")
        lines.append("- pulse_t, step_t = encoded from events (monthly)")
        lines.append("")
        lines.append("## Files produced (this run)")
        lines.append("- summary.json: parameters and fit metrics")
        lines.append("- fitted_series.csv: fitted values for overlay")
        lines.append("- covariates.csv: monthly ad_spend")
        lines.append("- features.csv: adstock and log ad effect")
        if isinstance(ev_out, pd.DataFrame) and not ev_out.empty:
            lines.append("- events_normalized.csv: events used in fit")

        (out_dir_path / "equation.md").write_text("\n".join(lines))
        logger.info("Equation saved: %s", str((out_dir_path / "equation.md").resolve()))
    except Exception:
        logger.info("Equation document not written")

    # Final accomplishment log
    # Final narrative summary in logs
    outputs_list = sorted([p.name for p in out_dir_path.iterdir()])
    logger.info("Purpose: ingest series + ads, build features, detect changes, fit growth model.")
    logger.info("Results: breakpoints=%s, K=%s, segments=%d", bkps, int(k_now) if k_now else None, len(r_list))
    logger.info("Saved: %s", ", ".join(outputs_list))
    logger.info(
        "Use: equation.md for formula + params; summary.json for structured params; "
        "features.csv provides exogenous x_t; fitted_series.csv for overlay."
    )


def _col_arg(s: str) -> str | int:
    # Helper: parse column arg as int if possible, else keep as string (name)
    try:
        return int(s)
    except (ValueError, TypeError):
        return s


def main() -> None:
    p = argparse.ArgumentParser(description="Headless runner for Substack Analyzer — Phase 1 (fit)")
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
    p.add_argument(
        "--detect-on",
        type=str,
        default="auto",
        choices=["auto", "total", "paid", "free", "both"],
        help="Which series to detect change points on (and merge if both)",
    )
    p.add_argument("--lam", type=float, default=0.5, help="Adstock lambda carryover [0,1)")
    p.add_argument("--theta", type=float, default=500.0, help="Ad effect log scale")
    p.add_argument("--out-dir", type=str, default="./outputs", help="Output directory")

    args = p.parse_args()

    cfg = PhaseOneConfig(
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
        detect_on=args.detect_on,
        lam=args.lam,
        theta=args.theta,
        out_dir=args.out_dir,
    )
    run_from_phase_one_config(cfg)


if __name__ == "__main__":
    main()
