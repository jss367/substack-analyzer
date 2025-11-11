from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from substack_analyzer.detection import detect_change_points

Effect = Literal["Transient", "Persistent", "No effect"]
Component = Literal["pulse", "level", "rate", "mixed", "none"]


@dataclass(frozen=True)
class BreakpointEffect:
    index: int
    date: pd.Timestamp
    effect: Effect
    component: Component
    slope_pre: float
    slope_post: float
    slope_delta: float
    jump_size: float  # Estimated level shift magnitude (post median - pre median)
    z_spike: float
    rate_score: float  # |Δslope| relative to rate threshold
    level_score: float  # |Δlevel| relative to level threshold
    note: str | None = None


@dataclass(frozen=True)
class DetectionConfig:
    """Shared configuration for change-point detection across app + headless runs."""

    use_classifier: bool = True
    max_changes: int = 4
    min_seg_len: int = 2
    penalty_scale: float = 4.0
    window: int = 6
    z_pulse: float = 3.0
    rate_factor: float = 0.75
    level_factor: float = 0.75


@dataclass(frozen=True)
class DetectionResult:
    """Container for the outputs of ``run_detection``."""

    indices: list[int]
    classified: list[BreakpointEffect]


def _mad(x: np.ndarray) -> float:
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m))


def _fit_line(y: pd.Series) -> tuple[float, float]:
    n = len(y)
    if n < 2:
        return 0.0, float(y.iloc[-1]) if n else 0.0
    x = np.arange(n, dtype=float)
    b, a = np.polyfit(x, y.to_numpy(dtype=float), deg=1)  # slope, intercept
    return float(b), float(a)


def classify_breakpoints_effect(
    input_series: pd.Series,
    candidates: List[int],
    window: int = 6,
    z_pulse: float = 3.0,
    rate_factor: float = 0.75,
    level_factor: float = 0.75,
) -> List[BreakpointEffect]:
    """Return effect (Transient/Persistent/No effect) and component (pulse/level/rate/mixed/none) per candidate."""
    input_series = input_series.dropna().sort_index()
    if len(input_series) < (2 * window + 2):
        return []
    ds = input_series.diff().dropna()

    global_sig_delta = _mad(ds.to_numpy()) or (np.std(ds.to_numpy(), ddof=1) or 1.0)
    global_sig_level = _mad(input_series.to_numpy()) or (np.std(input_series.to_numpy(), ddof=1) or 1.0)

    out: List[BreakpointEffect] = []
    for k in sorted(set(int(i) for i in candidates)):
        if k <= 0 or k >= len(input_series) - 1:
            continue
        pre_s = input_series.iloc[max(0, k - window) : k]
        post_s = input_series.iloc[k : min(len(input_series), k + window)]
        pre_d = ds.iloc[max(0, k - window) : k]
        post_d = ds.iloc[k : min(len(ds), k + window)]
        if len(pre_s) < 2 or len(post_s) < 2 or len(pre_d) < 1 or len(post_d) < 1:
            continue

        local_delta = np.concatenate((pre_d.to_numpy(dtype=float), post_d.to_numpy(dtype=float)))
        local_level = np.concatenate((pre_s.to_numpy(dtype=float), post_s.to_numpy(dtype=float)))

        sig_delta = _mad(local_delta) or (np.std(local_delta, ddof=1) or global_sig_delta)
        sig_level = _mad(local_level) or (np.std(local_level, ddof=1) or global_sig_level)
        if not np.isfinite(sig_delta) or sig_delta <= 0:
            sig_delta = global_sig_delta or 1.0
        if not np.isfinite(sig_level) or sig_level <= 0:
            sig_level = global_sig_level or 1.0
        if np.isfinite(global_sig_delta) and global_sig_delta > 0:
            sig_delta = min(sig_delta, global_sig_delta)
        if np.isfinite(global_sig_level) and global_sig_level > 0:
            sig_level = min(sig_level, global_sig_level)

        tau_rate = rate_factor * sig_delta
        tau_level = level_factor * sig_level

        mu_pre, mu_post = float(np.median(pre_d)), float(np.median(post_d))
        delta_mu = mu_post - mu_pre
        spike = float(input_series.iloc[k] - input_series.iloc[k - 1]) if k > 0 else 0.0
        z_spike = spike / (sig_delta or 1.0)

        slope_pre, a_pre = _fit_line(pre_s)
        slope_post, a_post = _fit_line(post_s)

        median_pre = float(np.median(pre_s))
        median_post = float(np.median(post_s))
        level_jump = median_post - median_pre
        slope_delta = slope_post - slope_pre

        # Decide effect & component
        effect: Effect = "No effect"
        component: Component = "none"
        note = None

        if abs(z_spike) >= z_pulse and abs(delta_mu) < 0.5 * tau_rate and abs(level_jump) < tau_level:
            effect, component = "Transient", "pulse"
            note = f"pulse z≈{z_spike:.1f}"
            rate_score = 0.0
            level_score = 0.0
        else:
            rate_flag = abs(slope_delta) >= tau_rate
            level_flag = abs(level_jump) >= tau_level
            rate_score = abs(slope_delta) / tau_rate if tau_rate > 0 else float("inf")
            level_score = abs(level_jump) / tau_level if tau_level > 0 else float("inf")
            if rate_flag and level_flag:
                effect, component = "Persistent", "mixed"
                note = f"Δslope={slope_delta:.3f}/mo; step≈{level_jump:.1f}"
            elif rate_flag:
                effect, component = "Persistent", "rate"
                note = f"Δslope={slope_delta:.3f}/mo"
            elif level_flag:
                effect, component = "Persistent", "level"
                note = f"step≈{level_jump:.1f}"
            else:
                effect, component = "No effect", "none"
                note = "weak/no change"
            if not np.isfinite(rate_score):
                rate_score = 0.0
            if not np.isfinite(level_score):
                level_score = 0.0

        out.append(
            BreakpointEffect(
                index=k,
                date=input_series.index[k].to_period("M").to_timestamp("M"),
                effect=effect,
                component=component,
                slope_pre=float(slope_pre),
                slope_post=float(slope_post),
                slope_delta=float(slope_delta),
                jump_size=float(level_jump),
                z_spike=float(z_spike),
                rate_score=float(rate_score),
                level_score=float(level_score),
                note=note,
            )
        )
    return out


def breakpoints_to_events(bps: List[BreakpointEffect], target_label: str) -> pd.DataFrame:
    rows = []
    for b in bps:
        if b.effect == "No effect":
            continue
        rows.append(
            {
                "date": b.date.date(),
                "type": "Other",  # keep your existing taxonomy
                "persistence": b.effect,
                "notes": f"{b.component}; {b.note} in {target_label}" if b.note else f"{b.component} in {target_label}",
                "cost": 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["date", "type", "persistence", "notes", "cost"])


def breakpoints_for_segments(bps: List[BreakpointEffect]) -> List[int]:
    def is_segment_worthy(b: BreakpointEffect) -> bool:
        if b.effect != "Persistent":
            return False
        if b.component in {"rate", "mixed"}:
            return True
        if b.component == "level" and b.level_score >= 1.5:
            return True
        return False

    return sorted(set(b.index for b in bps if is_segment_worthy(b)))


def filter_breakpoints(
    bps: Sequence[BreakpointEffect],
    *,
    effects: Iterable[Effect] | None = None,
    components: Iterable[Component] | None = None,
) -> list[BreakpointEffect]:
    """Return a filtered list of ``BreakpointEffect`` objects matching the criteria."""

    eff_set = {e for e in (effects or [])}
    comp_set = {c for c in (components or [])}

    def _matches(b: BreakpointEffect) -> bool:
        eff_ok = True if not eff_set else b.effect in eff_set
        comp_ok = True if not comp_set else b.component in comp_set
        return eff_ok and comp_ok

    return [b for b in bps if _matches(b)]


def run_detection(input_series: pd.Series, config: DetectionConfig | None = None) -> DetectionResult:
    """Run change-point detection with shared configuration.

    Parameters
    ----------
    input_series:
        Series to analyse (will be ``dropna``/``sort_index`` cleaned).
    config:
        Optional :class:`DetectionConfig`. When omitted the defaults mirror the
        interactive app (classifier-based detector with a six-month window).
    """

    cfg = config or DetectionConfig()
    series = input_series.dropna().sort_index()
    if series.empty:
        return DetectionResult(indices=[], classified=[])

    if cfg.use_classifier:
        classified = detect_and_classify(
            series,
            max_changes=cfg.max_changes,
            min_seg_len=cfg.min_seg_len,
            penalty_scale=cfg.penalty_scale,
            window=cfg.window,
            z_pulse=cfg.z_pulse,
            rate_factor=cfg.rate_factor,
            level_factor=cfg.level_factor,
        )
        indices = sorted({int(b.index) for b in classified})
        return DetectionResult(indices=indices, classified=classified)

    indices = list(
        detect_change_points(
            series,
            max_changes=cfg.max_changes,
            min_seg_len=cfg.min_seg_len,
            penalty_scale=cfg.penalty_scale,
            return_mode="indices",
        )
    )
    return DetectionResult(indices=sorted({int(i) for i in indices}), classified=[])


def detect_and_classify(
    input_series: pd.Series,
    *,
    # detection knobs (mirror current detector defaults)
    max_changes: int = 4,
    min_seg_len: int = 2,
    penalty_scale: float = 4.0,
    # classification knobs
    window: int = 6,
    z_pulse: float = 3.0,
    rate_factor: float = 0.75,
    level_factor: float = 0.75,
    # optional override: provide explicit candidates
    candidates: Optional[List[int]] = None,
) -> List[BreakpointEffect]:
    """Detect candidate change points and classify their effects/components in one call."""
    input_series = input_series.dropna().sort_index()
    if candidates is None:
        candidates = detect_change_points(
            input_series,
            max_changes=max_changes,
            min_seg_len=min_seg_len,
            penalty_scale=penalty_scale,
            return_mode="indices",
        )
    return classify_breakpoints_effect(
        input_series,
        candidates=candidates or [],
        window=window,
        z_pulse=z_pulse,
        rate_factor=rate_factor,
        level_factor=level_factor,
    )
