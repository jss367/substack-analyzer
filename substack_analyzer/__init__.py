try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("substack-analyzer")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

from substack_analyzer.model import simulate_growth
from substack_analyzer.types import AdSpendSchedule, SimulationInputs, SimulationResult

__all__ = [
    "__version__",
    "AdSpendSchedule",
    "SimulationInputs",
    "SimulationResult",
    "simulate_growth",
]
