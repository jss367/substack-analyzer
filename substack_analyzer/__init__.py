import tomllib
from pathlib import Path

from substack_analyzer.model import simulate_growth
from substack_analyzer.types import AdSpendSchedule, SimulationInputs, SimulationResult

__all__ = [
    "__version__",
    "AdSpendSchedule",
    "SimulationInputs",
    "SimulationResult",
    "simulate_growth",
]


pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
if pyproject.exists():
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    proj = data.get("project") or {}
    ver = proj.get("version")
    __version__ = ver.strip() if isinstance(ver, str) and ver.strip() else "0.0.0+local"
else:
    __version__ = "version not found"
