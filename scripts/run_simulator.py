"""
Headless Phase 2 simulator.

Reads Phase 1 outputs (summary.json) and runs a scenario simulation (e.g., constant
monthly ad spend) to project subscribers, revenue and profit. Writes a CSV with the
monthly results and logs a concise summary.
"""

import argparse
import json
import logging
import math
from pathlib import Path

import coloredlogs
import pandas as pd

from substack_analyzer.analysis import compute_estimates
from substack_analyzer.model import simulate_growth
from substack_analyzer.types import DEFAULT_GROWTH_RATE, AdSpendSchedule, SimulationInputs

coloredlogs.install(level="DEBUG")
logger = logging.getLogger("substack_simulator")


def _read_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found: {path}")
    return json.loads(path.read_text())


def _read_phase1(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"phase1.json not found: {path}")
    return json.loads(path.read_text())


def _records_to_series(records: list[dict] | None) -> pd.Series | None:
    if not records:
        return None
    df = pd.DataFrame(records)
    if not {"date", "count"}.issubset(df.columns):
        return None
    df = df.assign(date=lambda d: pd.to_datetime(d["date"]))
    df = df.dropna(subset=["date"]).sort_values("date")
    s = pd.to_numeric(df["count"], errors="coerce").dropna()
    if s.empty:
        return None
    s.index = df["date"].dt.to_period("M").dt.to_timestamp("M")
    return pd.Series(s.values, index=s.index)


def run(
    summary_path: str | None,
    phase1_path: str | None,
    from_out_dir: str | None,
    out_dir: str,
    spend_const: float | None,
    spend_stage1: float | None,
    spend_stage2: float | None,
    spend_once: float | None,
    once_month: int,
    horizon: int,
    cac: float,
    ad_fee: float,
    price_monthly: float,
    annual_share: float,
) -> None:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Inputs from Phase 1
    est: dict = {}
    fit_params: dict | None = None
    carrying_capacity: float | None = None
    if phase1_path:
        p1 = _read_phase1(Path(phase1_path))
        total_series = _records_to_series(p1.get("total_series"))
        paid_series = _records_to_series(p1.get("paid_series"))
        est = compute_estimates(all_series=total_series, paid_series=paid_series, window_months=6)
        logger.info("Reading Phase 1 (phase1.json): %s", str(Path(phase1_path).resolve()))
        fit_params = p1.get("fit_params") if isinstance(p1, dict) else None
        lam = float(p1.get("adstock_lambda", 0.5))
        theta = float(p1.get("ad_log_theta", 500.0))
        if fit_params and "carrying_capacity" in fit_params:
            try:
                carrying_capacity_val = float(fit_params.get("carrying_capacity"))
            except (TypeError, ValueError):
                carrying_capacity_val = 0.0
            if carrying_capacity_val > 0:
                carrying_capacity = carrying_capacity_val
    else:
        # Locate summary.json
        if summary_path:
            summary_file = Path(summary_path)
        elif from_out_dir:
            summary_file = Path(from_out_dir) / "summary.json"
        else:
            summary_file = out_dir_path / "summary.json"
        logger.info("Reading Phase 1 summary: %s", str(summary_file))
        summary = _read_summary(summary_file)
        est = summary.get("estimates", {}) or {}
        # Try to collect fit params when available in summary
        if all(k in summary for k in ("carrying_capacity", "segment_growth_rates")):
            fit_params = {
                "carrying_capacity": float(summary.get("carrying_capacity")),
                "segment_growth_rates": list(summary.get("segment_growth_rates", []) or []),
                "breakpoints": list(summary.get("breakpoints", []) or []),
                "gamma_pulse": float(summary.get("gamma_pulse", 0.0)),
                "gamma_step": float(summary.get("gamma_step", 0.0)),
                "gamma_exog": summary.get("gamma_exog"),
                "gamma_intercept": float(summary.get("gamma_intercept", 0.0)),
            }
            carrying_capacity = float(summary.get("carrying_capacity", 0.0)) or None
        lam = 0.5
        theta = 500.0

    # Seed from estimates; allow scenario-only run even if sparse
    start_free = int(est.get("start_free", 0))
    start_premium = int(est.get("start_premium", 0))
    organic_growth = float(est.get("organic_growth", DEFAULT_GROWTH_RATE))
    churn_free = float(est.get("churn_free", 0.0))
    churn_prem = float(est.get("churn_prem", 0.0))
    conv_ongoing = float(est.get("conv_ongoing", 0.0))

    # Spend schedule
    if spend_const is not None and spend_const > 0:
        schedule = AdSpendSchedule.constant(spend_const)
        sim_name = f"sim_const_{int(spend_const)}.csv"
    elif (spend_stage1 is not None and spend_stage2 is not None) and (spend_stage1 > 0 or spend_stage2 > 0):
        schedule = AdSpendSchedule.two_stage(spend_stage1 or 0.0, spend_stage2 or 0.0)
        sim_name = f"sim_two_stage_{int(spend_stage1 or 0)}_{int(spend_stage2 or 0)}.csv"
    elif spend_once is not None and spend_once > 0:
        # one-time investment at the specified month (1-based in CLI, convert to 0-indexed)
        schedule = AdSpendSchedule.one_time(spend_once, max(once_month - 1, 0))
        sim_name = f"sim_once_{int(spend_once)}_m{once_month}.csv"
    else:
        schedule = AdSpendSchedule.constant(0.0)
        sim_name = "sim_const_0.csv"

    inputs = SimulationInputs(
        starting_free_subscribers=start_free,
        starting_premium_subscribers=start_premium,
        carrying_capacity=carrying_capacity,
        horizon_months=horizon,
        organic_monthly_growth_rate=organic_growth,
        monthly_churn_rate_free=churn_free,
        monthly_churn_rate_premium=churn_prem,
        new_subscriber_premium_conv_rate=0.0,
        ongoing_premium_conv_rate=conv_ongoing,
        cost_per_new_free_subscriber=cac,
        ad_spend_schedule=schedule,
        ad_manager_monthly_fee=ad_fee,
        premium_monthly_price_gross=price_monthly,
        annual_share=annual_share,
    )

    logger.info(
        "Simulating: horizon=%d, CAC=%.2f, monthly_price=%.2f, annual_share=%.2f, schedule=%s",
        horizon,
        cac,
        price_monthly,
        annual_share,
        sim_name.replace(".csv", ""),
    )

    res = simulate_growth(inputs)
    out_file = out_dir_path / sim_name
    res.monthly.to_csv(out_file, index=False)

    s = res.summary
    logger.info(
        "Result: ending_total=%s, ending_premium=%s, cumulative_profit=%s, cumulative_ad_spend=%s",
        f"{int(s['ending_total']):,}",
        f"{int(s['ending_premium']):,}",
        f"${s['cumulative_net_profit']:,.0f}",
        f"${s['cumulative_ad_spend']:,.0f}",
    )
    # Payback month (first month cumulative_net_profit > 0)
    cum = pd.to_numeric(res.monthly["cumulative_net_profit"], errors="coerce")
    idx = next((i for i, v in enumerate(cum.tolist()) if float(v) > 0.0), None)
    if idx is None:
        logger.info("Payback: none within horizon (%d months)", horizon)
    else:
        logger.info("Payback: month %d", idx + 1)
    logger.info("Simulator output: %s", str(out_file.resolve()))

    # Optional: Forecast total subscribers using fitted equation if fit params were provided
    if fit_params is not None:
        K = float(carrying_capacity or fit_params.get("carrying_capacity", 0.0))
        r_list = [float(x) for x in (fit_params.get("segment_growth_rates") or [])]
        # Unused here: breakpoints and gamma_pulse/step (kept for compatibility)
        gamma_exog = fit_params.get("gamma_exog")
        gamma_intercept = float(fit_params.get("gamma_intercept", 0.0))
        exog_lag = int(fit_params.get("exog_lag")) if fit_params.get("exog_lag") is not None else 0

        # Determine starting S0 and last adstock
        S0 = float(est.get("start_free", 0) + est.get("start_premium", 0))
        last_adstock = 0.0
        if from_out_dir:
            fpath = Path(from_out_dir) / "features.csv"
            if fpath.exists():
                fdf = pd.read_csv(fpath)
                if "adstock" in fdf.columns:
                    last_adstock = float(pd.to_numeric(fdf["adstock"], errors="coerce").dropna().iloc[-1])

        # Build adstock ahead using schedule
        if spend_const is not None and spend_const > 0:
            schedule = AdSpendSchedule.constant(spend_const)
        elif (spend_stage1 is not None and spend_stage2 is not None) and (spend_stage1 > 0 or spend_stage2 > 0):
            schedule = AdSpendSchedule.two_stage(spend_stage1 or 0.0, spend_stage2 or 0.0)
        elif spend_once is not None and spend_once > 0:
            schedule = AdSpendSchedule.one_time(spend_once, max(once_month - 1, 0))
        else:
            schedule = AdSpendSchedule.constant(0.0)

        adstock_vals = []
        prev_a = last_adstock
        for m in range(horizon):
            x = float(schedule.get_spend_for_month(m))
            a = x + lam * prev_a
            adstock_vals.append(a)
            prev_a = a
        x_log = [0.0] * horizon
        if theta and theta > 0:
            x_log = [math.log(1.0 + a / theta) for a in adstock_vals]

        # Piecewise segment mapping for forecast months: use last segment rate
        r_last = float(r_list[-1] if r_list else 0.0)

        # Simulate
        S = [S0]
        for t in range(1, horizon + 1):
            S_prev = S[-1]
            x_base = S_prev * (1.0 - S_prev / K) if K > 0 else 0.0
            r_t = r_last
            delta = gamma_intercept + r_t * x_base
            if gamma_exog is not None:
                lag_idx = t - 1 - max(exog_lag, 0)
                if lag_idx >= 0 and lag_idx < len(x_log):
                    delta += float(gamma_exog) * float(x_log[lag_idx])
            S.append(max(S_prev + delta, 0.0))

        # Write forecast
        fc_index = pd.date_range(pd.Timestamp.today().to_period("M").to_timestamp("M"), periods=horizon, freq="ME")
        fc_df = pd.DataFrame(
            {"total_forecast": pd.Series(S[1:], index=fc_index).round().astype(int)},
            index=fc_index,
        )
        out_fit = Path(out_dir) / ("fitted_forecast.csv")
        fc_df.to_csv(out_fit, index_label="date")
        logger.info("Fitted-equation forecast saved: %s", str(out_fit.resolve()))


def _parse_phase2_config(p1: dict) -> dict:
    """
    Parse optional Phase 2 config embedded in phase1.json under key "phase2".

    Supported schema examples:
    {
      "phase2": {
        "horizon": 60,
        "cac": 2.0,
        "ad_fee": 0.0,
        "price_monthly": 10.0,
        "annual_share": 0.0,
        "schedule": {"type": "const", "amount": 5000}
      }
    }
    or
    {
      "phase2": {
        "horizon": 36,
        "schedule": {"type": "once", "amount": 1000, "once_month": 1}
      }
    }
    or
    {
      "phase2": {
        "horizon": 60,
        "schedule": {"type": "two_stage", "stage1": 5000, "stage2": 2000}
      }
    }
    """
    cfg = (p1 or {}).get("phase2") or {}
    horizon = int(cfg.get("horizon", 60))
    cac = float(cfg.get("cac", 2.0))
    ad_fee = float(cfg.get("ad_fee", 0.0))
    price_monthly = float(cfg.get("price_monthly", 10.0))
    annual_share = float(cfg.get("annual_share", 0.0))

    sched = cfg.get("schedule") or {}
    sched_type = (sched.get("type") or "const").lower()
    spend_const = spend_stage1 = spend_stage2 = spend_once = None
    once_month = int(sched.get("once_month", 1))
    if sched_type == "const":
        spend_const = float(sched.get("amount", 0.0))
    elif sched_type in ("one_time", "once"):
        spend_once = float(sched.get("amount", 0.0))
    elif sched_type in ("two_stage", "two-stage"):
        spend_stage1 = float(sched.get("stage1", 0.0))
        spend_stage2 = float(sched.get("stage2", 0.0))

    return {
        "horizon": horizon,
        "cac": cac,
        "ad_fee": ad_fee,
        "price_monthly": price_monthly,
        "annual_share": annual_share,
        "spend_const": spend_const,
        "spend_stage1": spend_stage1,
        "spend_stage2": spend_stage2,
        "spend_once": spend_once,
        "once_month": once_month,
    }


def run_with_phase1(phase1_path: str, out_dir: str, from_out_dir: str | None = None) -> None:
    """
    Convenience entrypoint: pass only a phase1.json path (with optional embedded
    "phase2" configuration) and an output directory. This avoids specifying a
    long list of parameters.
    """
    p1 = _read_phase1(Path(phase1_path))
    cfg = _parse_phase2_config(p1)

    # Delegate to main run() using parameters derived from phase1.json
    run(
        summary_path=None,
        phase1_path=phase1_path,
        from_out_dir=from_out_dir,
        out_dir=out_dir,
        spend_const=cfg["spend_const"],
        spend_stage1=cfg["spend_stage1"],
        spend_stage2=cfg["spend_stage2"],
        spend_once=cfg["spend_once"],
        once_month=cfg["once_month"],
        horizon=cfg["horizon"],
        cac=cfg["cac"],
        ad_fee=cfg["ad_fee"],
        price_monthly=cfg["price_monthly"],
        annual_share=cfg["annual_share"],
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Headless simulator — Phase 2 (scenario)")
    p.add_argument("--summary", dest="summary_path", type=str, default=None, help="Path to summary.json from Phase 1")
    p.add_argument("--phase1", dest="phase1_path", type=str, default=None, help="Path to phase1.json artifact")
    p.add_argument(
        "--from-out-dir", dest="from_out_dir", type=str, default=None, help="Directory that contains summary.json"
    )
    p.add_argument("--out-dir", type=str, default="./outputs", help="Output directory")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--spend-const", dest="spend_const", type=float, default=None, help="Constant monthly ad spend")
    g.add_argument(
        "--spend-two-stage",
        nargs=2,
        metavar=("STAGE1", "STAGE2"),
        type=float,
        default=None,
        help="Two-stage spend (year 1 and year 2+)",
    )
    g.add_argument(
        "--spend-once", dest="spend_once", type=float, default=None, help="One-time ad spend in a single month"
    )
    p.add_argument("--once-month", type=int, default=1, help="Month number (1-based) to apply one-time spend")

    p.add_argument("--horizon", type=int, default=60, help="Months to simulate")
    p.add_argument("--cac", type=float, default=2.0, help="Cost per new free subscriber (CAC)")
    p.add_argument("--ad-fee", type=float, default=0.0, help="Ad manager monthly fee")
    p.add_argument("--price-monthly", type=float, default=10.0, help="Premium monthly price (gross)")
    p.add_argument("--annual-share", type=float, default=0.0, help="Share of premium on annual plans (0..1)")

    args = p.parse_args()

    spend_stage1 = spend_stage2 = None
    if args.spend_two_stage is not None:
        spend_stage1 = args.spend_two_stage[0]
        spend_stage2 = args.spend_two_stage[1]

    run(
        summary_path=args.summary_path,
        phase1_path=args.phase1_path,
        from_out_dir=args.from_out_dir,
        out_dir=args.out_dir,
        spend_const=args.spend_const,
        spend_stage1=spend_stage1,
        spend_stage2=spend_stage2,
        spend_once=args.spend_once,
        once_month=args.once_month,
        horizon=args.horizon,
        cac=args.cac,
        ad_fee=args.ad_fee,
        price_monthly=args.price_monthly,
        annual_share=args.annual_share,
    )


if __name__ == "__main__":
    main()
