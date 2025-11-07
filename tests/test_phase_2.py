import importlib.util
from pathlib import Path

import pandas as pd


def _load_run_simulator() -> object:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_simulator.py"
    spec = importlib.util.spec_from_file_location("run_simulator", str(script_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Failed to load run_simulator module spec"
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def test_phase2_constant_spend_file_only(tmp_path: Path):
    run_sim = _load_run_simulator()
    out_dir = tmp_path
    fixtures = _fixtures_dir()

    run_sim.run_with_phase1(
        phase1_path=str(fixtures / "phase1_const.json"),
        out_dir=str(out_dir),
        from_out_dir=None,
    )

    out_file = out_dir / "sim_const_5000.csv"
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert len(df) == 60
    assert df["cumulative_ad_spend"].iloc[-1] == 60 * 5000

    assert (out_dir / "fitted_forecast.csv").exists()


def test_phase2_one_time_payback_file_only(tmp_path: Path):
    run_sim = _load_run_simulator()
    out_dir = tmp_path
    fixtures = _fixtures_dir()

    run_sim.run_with_phase1(
        phase1_path=str(fixtures / "phase1_once.json"),
        out_dir=str(out_dir),
        from_out_dir=None,
    )

    out_file = out_dir / "sim_once_1000_m1.csv"
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert len(df) == 36
    assert df["cumulative_ad_spend"].iloc[-1] == 1000


def test_phase2_minimal_with_phase1_file_only(tmp_path: Path):
    run_sim = _load_run_simulator()
    out_dir = tmp_path
    fixtures = _fixtures_dir()

    run_sim.run_with_phase1(
        phase1_path=str(fixtures / "phase1.json"),
        out_dir=str(out_dir),
        from_out_dir=None,
    )

    out_file = out_dir / "sim_once_1000_m1.csv"
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert len(df) == 36

    assert (out_dir / "fitted_forecast.csv").exists()
