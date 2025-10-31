"""
Headless Phase 2 simulator.

Reads Phase 1 outputs (summary.json) and runs a scenario simulation (e.g., constant
monthly ad spend) to project subscribers, revenue and profit. Writes a CSV with the
monthly results and logs a concise summary.
"""

import argparse
import json
import logging
from pathlib import Path

import coloredlogs

from substack_analyzer.model import simulate_growth
from substack_analyzer.types import AdSpendSchedule, SimulationInputs

coloredlogs.install(level="DEBUG")
logger = logging.getLogger("substack_simulator")


def _read_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found: {path}")
    return json.loads(path.read_text())


def run(
    summary_path: str | None,
    from_out_dir: str | None,
    out_dir: str,
    spend_const: float | None,
    spend_stage1: float | None,
    spend_stage2: float | None,
    horizon: int,
    cac: float,
    ad_fee: float,
    price_monthly: float,
    annual_share: float,
) -> None:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

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

    # Seed from estimates; allow scenario-only run even if sparse
    start_free = int(est.get("start_free", 0))
    start_premium = int(est.get("start_premium", 0))
    organic_growth = float(est.get("organic_growth", 0.01))
    churn_free = float(est.get("churn_free", 0.0))
    churn_prem = float(est.get("churn_prem", 0.0))
    conv_ongoing = float(est.get("conv_ongoing", 0.0))

    # Spend schedule
    if spend_const is not None and spend_const > 0:
        schedule = AdSpendSchedule.constant(float(spend_const))
        sim_name = f"sim_const_{int(spend_const)}.csv"
    elif (spend_stage1 is not None and spend_stage2 is not None) and (spend_stage1 > 0 or spend_stage2 > 0):
        schedule = AdSpendSchedule.two_stage(float(spend_stage1 or 0.0), float(spend_stage2 or 0.0))
        sim_name = f"sim_two_stage_{int(spend_stage1 or 0)}_{int(spend_stage2 or 0)}.csv"
    else:
        schedule = AdSpendSchedule.constant(0.0)
        sim_name = "sim_const_0.csv"

    inputs = SimulationInputs(
        starting_free_subscribers=start_free,
        starting_premium_subscribers=start_premium,
        horizon_months=int(horizon),
        organic_monthly_growth_rate=organic_growth,
        monthly_churn_rate_free=churn_free,
        monthly_churn_rate_premium=churn_prem,
        new_subscriber_premium_conv_rate=0.0,
        ongoing_premium_conv_rate=conv_ongoing,
        cost_per_new_free_subscriber=float(cac),
        ad_spend_schedule=schedule,
        ad_manager_monthly_fee=float(ad_fee),
        premium_monthly_price_gross=float(price_monthly),
        annual_share=float(annual_share),
    )

    logger.info(
        "Simulating: horizon=%d, CAC=%.2f, monthly_price=%.2f, annual_share=%.2f, schedule=%s",
        int(horizon),
        float(cac),
        float(price_monthly),
        float(annual_share),
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
    logger.info("Simulator output: %s", str(out_file.resolve()))


def _col_arg(s: str) -> str | int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return s


def main() -> None:
    p = argparse.ArgumentParser(description="Headless simulator — Phase 2 (scenario)")
    p.add_argument("--summary", dest="summary_path", type=str, default=None, help="Path to summary.json from Phase 1")
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
        help="Two-stage spend (years 1-2, years 3-5)",
    )

    p.add_argument("--horizon", type=int, default=60, help="Months to simulate")
    p.add_argument("--cac", type=float, default=2.0, help="Cost per new free subscriber (CAC)")
    p.add_argument("--ad-fee", type=float, default=0.0, help="Ad manager monthly fee")
    p.add_argument("--price-monthly", type=float, default=10.0, help="Premium monthly price (gross)")
    p.add_argument("--annual-share", type=float, default=0.0, help="Share of premium on annual plans (0..1)")

    args = p.parse_args()

    spend_stage1 = spend_stage2 = None
    if args.spend_two_stage is not None:
        spend_stage1 = float(args.spend_two_stage[0])
        spend_stage2 = float(args.spend_two_stage[1])

    run(
        summary_path=args.summary_path,
        from_out_dir=args.from_out_dir,
        out_dir=args.out_dir,
        spend_const=args.spend_const,
        spend_stage1=spend_stage1,
        spend_stage2=spend_stage2,
        horizon=args.horizon,
        cac=args.cac,
        ad_fee=args.ad_fee,
        price_monthly=args.price_monthly,
        annual_share=args.annual_share,
    )


if __name__ == "__main__":
    main()
