import math
import numbers
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.logger import get_logger

from substack_analyzer import __version__
from substack_analyzer.analysis import (
    DEFAULT_AD_LOG_THETA,
    DEFAULT_ADSTOCK_LAMBDA,
    build_events_features,
    compute_estimates,
    derive_adds_churn,
    plot_series,
    read_series,
)
from substack_analyzer.calibration import fit_piecewise_logistic, fitted_series_from_params, forecast_piecewise_logistic
from substack_analyzer.changepoints import (
    BreakpointEffect,
    DetectionConfig,
    DetectionResult,
    breakpoints_to_events,
    filter_breakpoints,
    run_detection,
)
from substack_analyzer.detection import compute_segment_slopes, slope_around
from substack_analyzer.model import simulate_growth
from substack_analyzer.persistence import (
    apply_phase_one_json,
    apply_session_bundle,
    collect_session_bundle,
    export_phase_one_json,
)
from substack_analyzer.types import DEFAULT_GROWTH_RATE, AdSpendSchedule, SimulationInputs
from substack_analyzer.ui import format_currency as ui_format_currency
from substack_analyzer.ui import format_date_badges as ui_format_date_badges
from substack_analyzer.ui import inject_brand_styles as ui_inject_brand_styles
from substack_analyzer.ui import render_brand_header as ui_render_brand_header
from substack_analyzer.utils import coerce_list

# Asset paths
ASSETS_DIR = Path(__file__).parent / "logos"
LOGO_ICON = ASSETS_DIR / "ROPI_IconDark Green_RGB.png"
LOGO_FULL = ASSETS_DIR / "RPI_Full logo_Dark Green_RGB.png"

# MUST be the first Streamlit call:
st.set_page_config(
    page_title="Substack Ads ROI Simulator",
    layout="wide",
    page_icon=str(LOGO_ICON) if LOGO_ICON.exists() else (str(LOGO_FULL) if LOGO_FULL.exists() else None),
)

# Streamlit logger (appears in deployment logs)
logger = get_logger(__name__)
logger.info("App startup: version=%s", __version__)


# --- Events table: single source of truth ---
EVENTS_COLUMNS = ["date", "type", "persistence", "notes", "cost"]
TYPE_TO_PERSISTENCE = {
    "ad spend": "Transient",
    "ad": "Transient",
    "shout-out": "Transient",
    "viral post": "Transient",
    "launch": "Persistent",
    "paywall change": "Persistent",
    "change": "Transient",
    "other": None,
}
EVENT_TYPE_OPTIONS = [
    "Ad spend",
    "Shout-out",
    "Viral post",
    "Launch",
    "Paywall change",
    "Other",
]
if "events_df" not in st.session_state:
    st.session_state["events_df"] = pd.DataFrame(columns=EVENTS_COLUMNS)

STAGE_PHASES = [
    {
        "title": "Phase 1 — Fit & diagnostics",
        "summary": (
            "This phase happens on the Data Import tab. It cleans Substack exports, annotates events,"
            " derives heuristic adds and churn, fits the piecewise logistic model, and prepares a `phase1.json` handoff."
        ),
        "stages": [
            {
                "label": "Stage 1 — Import & normalization",
                "details": [
                    "Upload All and Paid subscriber exports (CSV/XLSX) with selectable columns.",
                    "Resample to month-end, derive Free = Total − Paid, and create `observations_df` for download.",
                    "Store the cleaned table in session state for downstream stages and `phase1.json` exports.",
                ],
            },
            {
                "label": "Stage 2 — Events & features",
                "details": [
                    "Use the editable event grid with change-point detection assists plus cost/notes metadata.",
                    "Build `events_df`, `covariates_df`, and `features_df` using adstock and log-response parameters from the sidebar.",
                    "Import or export `phase1.json` to hand off breakpoints, events, and ad spend.",
                ],
            },
            {
                "label": "Stage 3 — Adds & churn",
                "details": [
                    "Derive monthly `adds_df` and `churn_df` from totals using user-supplied churn-rate estimates.",
                    "Download CSVs for both tables while exploring latent adds/churn decomposition options.",
                ],
            },
            {
                "label": "Stage 4 — Quick fit",
                "details": [
                    "Fit a piecewise logistic model with detected breakpoints and optional ad-response regressor.",
                    "Expose carrying capacity, per-segment growth rates, and event coefficients with override sliders.",
                    "Export fitted overlays, forward projections, and the growth equation for simulation.",
                ],
            },
            {
                "label": "Stage 5 — Diagnostics",
                "details": [
                    "Review delta charts plus tail views with segment slope overlays and trailing-window metrics.",
                    "Surface quick estimators, support `phase1.json` downloads, and apply estimates to the Simulator sidebar.",
                ],
            },
        ],
    },
    {
        "title": "Phase 2 — Simulation & outputs",
        "summary": (
            "This phase lives on the Simulator and Save / Load tabs. It runs the deterministic cohort model and"
            " packages results for planning or hand-off."
        ),
        "stages": [
            {
                "label": "Stage 6 — Cohort & finance simulator",
                "details": [
                    "Run the deterministic free/paid cohort model with growth, churn, conversion, and ad-spend schedules.",
                    "Output monthly KPIs, charts, ROAS/CAC/payback tiles, and apply completed Phase 1 stage results.",
                    "Support two-stage versus constant spend, manual overrides, and stateful save/load bundles.",
                ],
            },
            {
                "label": "Stage 7 — Outputs & documentation",
                "details": [
                    "Export `phase1.json` and full session bundles (.zip) plus CSV downloads for intermediate tables.",
                    "Capture your current configuration for documentation or collaboration.",
                ],
            },
        ],
    },
]

DEFAULT_SIMULATOR_EQUATION_LATEX = (
    r"F_t = F_{t-1}(1 - c_f) + F_{t-1}\,g + \frac{AdSpend_t}{CAC} - conv_t\\"
    r"P_t = P_{t-1}(1 - c_p) + conv_t\\"
    r"conv_t = (new^{free}_t)\,p_{new} + F_{t-1}\,p_{ongoing},\\"
    r"\quad new^{free}_t = F_{t-1}\,g + \frac{AdSpend_t}{CAC}"
)


def _clean_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types without changing the date *month/day* a user entered."""
    logger.info("_clean_events_df has been called")
    logger.info("df: %s", df)
    df = df.copy()
    for col in EVENTS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    logger.info("df after adding missing columns: %s", df)
    # Coerce types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    # Fill persistence only where missing
    typed_lower = df.get("type").astype(str).str.lower()
    need_fill = df.get("persistence").isna() | (df.get("persistence").astype(str).str.len() == 0)
    df.loc[need_fill, "persistence"] = typed_lower.map(TYPE_TO_PERSISTENCE)
    logger.info("df at the end of _clean_events_df: %s", df)
    return df


def _event_rules_from_events() -> alt.Chart | None:
    logger.info("_event_rules_from_events has been called")
    ev = st.session_state.get("events_df")
    logger.info("ev: %s", ev)
    if not isinstance(ev, pd.DataFrame) or ev.empty or "date" not in ev.columns:
        return None
    ev2 = ev.copy()
    ev2["date"] = pd.to_datetime(ev2["date"], errors="coerce")
    ev2 = ev2.dropna(subset=["date"])
    # For visualization only: if an event lands exactly on month-end (typical for detected breakpoints),
    # show the vertical rule at the last month of the old segment (previous month-end).
    with suppress(Exception):
        ev2["marker_date"] = ev2["date"]
        mask_me = ev2["date"].dt.is_month_end.fillna(False)
        ev2.loc[mask_me, "marker_date"] = ev2.loc[mask_me, "date"] - pd.offsets.MonthEnd(1)
    # Normalize Effect labels for reliable styling
    eff_map = {"persistent": "Persistent", "transient": "Transient", "no effect": "No effect"}
    with suppress(Exception):
        ev2["effect_norm"] = ev2.get("persistence").astype(str).str.strip().str.lower().map(eff_map).fillna("Transient")

    layers = []
    # Persistent: green solid
    ev_p = ev2[ev2.get("effect_norm") == "Persistent"]
    if not ev_p.empty:
        logger.info("Adding persistent event rules")
        layers.append(
            alt.Chart(ev_p)
            .mark_rule(strokeWidth=2, color="#27ae60")
            .encode(
                x=alt.X("marker_date:T", title="Date"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("type:N", title="Type"),
                    alt.Tooltip("effect_norm:N", title="Effect"),
                    alt.Tooltip("notes:N", title="Notes"),
                    alt.Tooltip("cost:Q", title="Cost ($)"),
                ],
            )
        )
    # Transient: purple dashed
    ev_t = ev2[ev2.get("effect_norm") == "Transient"]
    if not ev_t.empty:
        logger.info("Adding transient event rules")
        layers.append(
            alt.Chart(ev_t)
            .mark_rule(strokeWidth=2, color="#8e44ad", strokeDash=[6, 4])
            .encode(
                x=alt.X("marker_date:T", title="Date"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("type:N", title="Type"),
                    alt.Tooltip("effect_norm:N", title="Effect"),
                    alt.Tooltip("notes:N", title="Notes"),
                    alt.Tooltip("cost:Q", title="Cost ($)"),
                ],
            )
        )
    # No effect: grey dotted
    ev_n = ev2[ev2.get("effect_norm") == "No effect"]
    if not ev_n.empty:
        logger.info("Adding no effect event rules")
        layers.append(
            alt.Chart(ev_n)
            .mark_rule(strokeWidth=2, color="#bdc3c7", strokeDash=[2, 4])
            .encode(
                x=alt.X("marker_date:T", title="Date"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("type:N", title="Type"),
                    alt.Tooltip("effect_norm:N", title="Effect"),
                    alt.Tooltip("notes:N", title="Notes"),
                    alt.Tooltip("cost:Q", title="Cost ($)"),
                ],
            )
        )

    return alt.layer(*layers) if layers else None


def _on_events_editor_change():
    logger.info("_on_events_editor_change has been called")
    grid_dict = st.session_state.get("events_editor") or {}
    logger.info("grid_dict: %s", grid_dict)
    with suppress(Exception):
        # Start from current events_df as the source of truth
        base = st.session_state.get("events_df", pd.DataFrame(columns=EVENTS_COLUMNS)).copy()

        if isinstance(grid_dict, dict):
            # Apply in-place cell edits
            edited = grid_dict.get("edited_rows") or {}
            for row_idx, changes in edited.items():
                try:
                    i = int(row_idx)
                except Exception:
                    continue
                for col, val in (changes or {}).items():
                    if col not in base.columns:
                        base[col] = None
                    if 0 <= i < len(base.index):
                        base.at[base.index[i], col] = val

            # Append any newly added rows
            added = grid_dict.get("added_rows") or []
            if isinstance(added, list) and added:
                base = pd.concat([base, pd.DataFrame(added)], ignore_index=True)

            # Remove any deleted rows by positional index (reverse order)
            deleted = grid_dict.get("deleted_rows") or []
            if isinstance(deleted, list) and deleted:
                drop_idx = sorted([int(x) for x in deleted if str(x).isdigit()], reverse=True)
                for i in drop_idx:
                    if 0 <= i < len(base.index):
                        base = base.drop(base.index[i])
        else:
            # Fallback: try to coerce whatever we have into a DataFrame
            base = pd.DataFrame(grid_dict)

        st.session_state["events_df"] = _clean_events_df(base)
        logger.info("st.session_state['events_df']: %s", st.session_state['events_df'])
        _set_markers_from_events()


def _events_change_dates() -> list[pd.Timestamp]:
    ev = st.session_state.get("events_df")
    if not isinstance(ev, pd.DataFrame) or ev.empty or "date" not in ev.columns:
        return []
    ev2 = ev.copy()
    ev2["date"] = pd.to_datetime(ev2["date"], errors="coerce")
    # Determine which event types count as breakpoints
    mode = str(st.session_state.get("breakpoint_mode", "all")).lower()
    types_series = ev2.get("type").astype(str).str.lower()
    effect = ev2.get("persistence").astype(str).str.lower()
    if mode == "all":
        mask = ev2["date"].notna() & ~effect.eq("no effect")
    elif mode == "selected":
        sel = st.session_state.get("breakpoint_types", [])
        sel_l = {str(t).lower() for t in (sel or [])}
        mask = types_series.isin(sel_l) & ~effect.eq("no effect")
    else:
        # throw error
        raise ValueError("Invalid breakpoint mode")
    return [pd.Timestamp(d) for d in ev2.loc[mask, "date"].dropna().tolist()]


def _set_markers_from_events() -> None:
    """Make the chart markers come from events (and keep them there)."""
    st.session_state["markers_source"] = "events"
    st.session_state["detected_change_dates"] = _events_change_dates()


def _normalize_month_end(dates: list[Any]) -> list[pd.Timestamp]:
    """Coerce arbitrary date-like values to month-end pd.Timestamp and de-duplicate/sort."""
    if dates is None:
        return []
    try:
        s = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
        s = s.dropna().dt.to_period("M").dt.to_timestamp("M")
        vals = [pd.Timestamp(d) for d in pd.unique(s.sort_values())]
        return vals
    except Exception:
        return []


def _dates_to_breakpoint_indices(dates: list[pd.Timestamp], index: pd.DatetimeIndex) -> list[int]:
    """Map month-end dates to integer indices into the given monthly index."""
    if dates is None or len(dates) == 0:
        return []
    idxs: list[int] = []
    for d in _normalize_month_end(dates):
        try:
            if d in index:
                idxs.append(int(index.get_loc(d)))
        except Exception:
            continue
    # unique + sorted + valid interior indices only
    return sorted({i for i in idxs if i is not None})


def _inject_brand_styles() -> None:
    ui_inject_brand_styles()


def render_brand_header() -> None:
    ui_render_brand_header(LOGO_FULL, LOGO_ICON)


def format_currency(value: float) -> str:
    return ui_format_currency(value)


def _get_state(key: str, default):
    return st.session_state.get(key, default)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    """Coerce model parameters to float without raising when None or invalid."""

    try:
        if value is None:
            return float(fallback)
        if isinstance(value, float) and math.isnan(value):
            return float(fallback)
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _apply_pending_state_updates() -> None:
    """Apply any deferred session state updates before widgets render."""

    pending = st.session_state.pop("_pending_state_update", None)
    if isinstance(pending, dict):
        for k, v in pending.items():
            st.session_state[k] = v


# Apply brand styles and sidebar logo once
_inject_brand_styles()
_apply_pending_state_updates()
if LOGO_FULL.exists() or LOGO_ICON.exists():
    st.sidebar.image(str(LOGO_FULL if LOGO_FULL.exists() else LOGO_ICON), width="stretch")


def read_head_preview(fh, has_header: bool, nrows: int = 5) -> pd.DataFrame:
    """Read a small preview from an uploaded CSV/XLSX without consuming the file pointer."""
    try:
        if fh.name.lower().endswith((".xlsx", ".xls")):
            tmp = pd.read_excel(fh, header=0 if has_header else None, nrows=nrows)
        else:
            tmp = pd.read_csv(fh, header=0 if has_header else None, nrows=nrows)
    finally:
        with suppress(Exception):
            fh.seek(0)
    return tmp


def upload_panel(
    title: str,
    help_hint: str,
    key_prefix: str,
    default_header: bool = False,
) -> tuple[Any | None, bool, int | None, int | None]:
    """Shared UI for file upload + optional header + column choices.

    Returns
    -------
    (file_obj, has_header, date_sel, count_sel)
        - file_obj: The uploaded file-like object, or None if no file selected or preview failed.
        - has_header: Whether the uploaded file is expected to contain a header row.
        - date_sel: The zero-based index of the date column derived from a small preview, or None.
        - count_sel: The zero-based index of the count column derived from a small preview, or None.

    Notes
    -----
    - The two selectboxes (date/count column indices) are only shown when a file is present and a
      preview can be read. If reading the preview fails, an error is shown and file_obj is set to None.
    - When no file is provided, the returned indices may be defaults and should be treated as optional.
    """
    file_obj = st.file_uploader(title, type=["csv", "xlsx", "xls"], key=f"{key_prefix}_file", help=help_hint)
    has_header = st.checkbox(
        f"{key_prefix.capitalize()} file has header row", value=default_header, key=f"{key_prefix}_has_header"
    )
    date_sel: int | None = 0
    count_sel: int | None = 1
    if file_obj is not None:
        try:
            head = read_head_preview(file_obj, has_header, nrows=5)
            ncols = head.shape[1]
            date_sel = st.selectbox(
                f"{key_prefix.capitalize()}: date column (index)",
                list(range(ncols)),
                index=0,
                key=f"{key_prefix}_date_sel",
            )
            count_sel = st.selectbox(
                f"{key_prefix.capitalize()}: count column (index)",
                list(range(ncols)),
                index=min(1, max(ncols - 1, 0)),
                key=f"{key_prefix}_count_sel",
            )
        except Exception as e:
            st.error(f"Could not read {key_prefix.capitalize()} file: {e}")
            file_obj = None
    return file_obj, has_header, date_sel, count_sel


def emit_observations(plot_df: pd.DataFrame) -> None:
    """Stage 1 output: observations_df (current granularity)."""
    idx = plot_df.index
    total = plot_df.get("Total")
    paid = plot_df.get("Paid")
    if total is not None and paid is not None:
        free = (total.astype(float) - paid.astype(float)).clip(lower=0)
    elif total is not None:
        free = total.astype(float) - float(_get_state("start_premium", 0))
    else:
        free = pd.Series(index=idx, dtype=float)

    obs = pd.DataFrame(
        {
            "active_total": (total.astype(float) if total is not None else pd.Series(index=idx, dtype=float)),
            "active_paid": (paid.astype(float) if paid is not None else pd.Series(index=idx, dtype=float)),
            "active_free": free.astype(float),
            "is_imputed": False,
        },
        index=idx,
    )
    obs.index.name = "date"
    st.session_state["observations_df"] = obs
    with st.expander("Stage 1 output: observations_df", expanded=False):
        st.dataframe(obs.reset_index(), width="stretch")
        st.download_button(
            "Download observations.csv",
            data=obs.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="observations.csv",
            mime="text/csv",
        )


def trend_detection_ui(plot_df: pd.DataFrame, target_col: str | None) -> list[int]:
    st.caption("Detection runs when you click the Events button above.")
    if target_col is None:
        return []
    # Let users choose detection sensitivity ahead of time; used when button is clicked.
    st.slider("Max changes to detect", 0, 8, 3, 1, key="max_changes_detect")

    # Show any previously detected results (if the user clicked the button).
    bkps = list(st.session_state.get("detected_breakpoints", []))
    if bkps:
        s_idx = plot_df[target_col].dropna().index
        dates = [pd.to_datetime(s_idx[i]) for i in bkps if i < len(s_idx)]
        used = st.session_state.get("detected_target_label", target_col)
        st.markdown(f"**Detected change dates (on {used}):**")
        st.markdown(ui_format_date_badges(dates), unsafe_allow_html=True)
    return bkps


def events_editor(plot_df: pd.DataFrame, target_col: str | None) -> None:
    st.subheader("Stage 2: Events & annotations")
    st.caption("Track shout-outs, ad campaigns, launches, etc. Dates must match the series timeline.")

    # Detection mode selector (store canonical code in session_state['detect_on'])
    _options = ["Both (Total→Paid)", "Both (Total+Paid)", "Auto (Total→Free)", "Total", "Free", "Paid"]
    # Backward compatible label set (some users may still see arrow variants); pick the first valid display for current code
    _label_to_code = {
        "Both (Total+Paid)": "both",
        "Both (Total→Paid)": "both",
        "Auto (Total→Free)": "auto",
        "Total": "total",
        "Free": "free",
        "Paid": "paid",
    }
    _code_to_label = {
        "both": "Both (Total+Paid)",
        "auto": "Auto (Total→Free)",
        "total": "Total",
        "free": "Free",
        "paid": "Paid",
    }
    _current_code = str(st.session_state.get("detect_on", "both")).lower()
    _current_label = _code_to_label.get(_current_code, "Both (Total+Paid)")
    try:
        _index = _options.index(_current_label) if _current_label in _options else 0
    except Exception:
        _index = 0
    _selected_label = st.selectbox(
        "Detection target",
        _options,
        index=_index,
        help=(
            "Choose which series to run change-point detection on. "
            "Auto prefers Total (or Free if Total unavailable). Both will detect on Total and Paid and merge."
        ),
        key="detect_on_display",
    )
    detect_mode = _label_to_code.get(_selected_label, "both")
    st.session_state["detect_on"] = detect_mode

    # Add detected change dates
    with st.container():
        add_col1, _ = st.columns([1, 3])
        with add_col1:
            if st.button("Detect change dates"):
                if target_col is None:
                    st.info("No target series selected for detection.")
                else:
                    max_changes = int(st.session_state.get("max_changes_detect", 3))

                    detect_cfg = DetectionConfig(use_classifier=True, max_changes=max_changes, window=6)

                    def _detect(series: pd.Series) -> DetectionResult:
                        return run_detection(series, config=detect_cfg)

                    # Debug: record detection configuration
                    with suppress(Exception):
                        logger.info(
                            "Detection clicked: mode=%s, max_changes=%s, target_col=%s, plot_cols=%s",
                            detect_mode,
                            max_changes,
                            target_col,
                            list(plot_df.columns),
                        )

                    def _log_classified(label: str, classified: list) -> None:
                        with suppress(Exception):
                            rows = [
                                {
                                    "index": int(getattr(b, "index", -1)),
                                    "date": str(getattr(b, "date", "")),
                                    "effect": str(getattr(b, "effect", "")),
                                    "component": str(getattr(b, "component", "")),
                                    "rate_score": float(getattr(b, "rate_score", 0.0)),
                                    "level_score": float(getattr(b, "level_score", 0.0)),
                                    "note": str(getattr(b, "note", "")),
                                }
                                for b in (classified or [])
                            ]
                            logger.info("Detection raw (%s): %s", label, rows)

                    # Determine which series to run on
                    detections: list[DetectionResult] = []
                    label_list: list[str] = []
                    # Auto
                    if detect_mode == "auto":
                        s_auto = plot_df[target_col].dropna()
                        detections = [_detect(s_auto)]
                        label_list = [target_col]
                        _log_classified(label_list[0], detections[0].classified)
                    # Explicit targets
                    elif detect_mode == "total" and ("Total" in plot_df.columns):
                        detections = [_detect(plot_df["Total"])]
                        label_list = ["Total"]
                        target_col = "Total"
                        _log_classified("Total", detections[0].classified)
                    elif detect_mode == "free" and ("Free" in plot_df.columns):
                        detections = [_detect(plot_df["Free"])]
                        label_list = ["Free"]
                        target_col = "Free"
                        _log_classified("Free", detections[0].classified)
                    elif detect_mode == "paid" and ("Paid" in plot_df.columns):
                        detections = [_detect(plot_df["Paid"])]
                        label_list = ["Paid"]
                        target_col = "Paid"
                        _log_classified("Paid", detections[0].classified)
                    elif detect_mode == "both" and ({"Total", "Paid"}.issubset(plot_df.columns)):
                        detections = [_detect(plot_df["Total"]), _detect(plot_df["Paid"])]
                        label_list = ["Total", "Paid"]
                        _log_classified("Total", detections[0].classified)
                        _log_classified("Paid", detections[1].classified)
                    else:
                        st.info("Requested detection target not available in current data.")
                        detections = []

                    # Merge results
                    merged_events_df = None
                    merged_classified: list[BreakpointEffect] = []

                    if not detections or all(len(res.classified) == 0 for res in detections):
                        st.info("No change dates detected with current settings.")
                    else:
                        # Seed events per source label
                        for result, label in zip(detections, label_list):
                            cls = result.classified
                            if not cls:
                                continue
                            seeded_df = breakpoints_to_events(cls, target_label=label)
                            merged_events_df = (
                                seeded_df
                                if merged_events_df is None
                                else pd.concat(
                                    [merged_events_df, seeded_df],
                                    ignore_index=True,
                                )
                            )
                            merged_classified.extend(cls)

                        # De-duplicate events and store
                        if merged_events_df is not None and not merged_events_df.empty:
                            # When detecting on both series, collapse duplicates by (date, persistence)
                            # to avoid two rows for the same month stemming from Total vs Paid.
                            if detect_mode == "both":
                                merged_events_df = merged_events_df.sort_values("date")
                                merged_events_df = merged_events_df.drop_duplicates(
                                    subset=["date", "persistence"], keep="first"
                                )
                            merged_events_df = merged_events_df.drop_duplicates(
                                subset=["date", "type", "notes"], keep="first"
                            )
                            base = st.session_state.get("events_df", pd.DataFrame(columns=EVENTS_COLUMNS))
                            merged_all = (
                                merged_events_df
                                if base.empty
                                else pd.concat([base, merged_events_df], ignore_index=True)
                            )
                            # Also de-duplicate across the combined table to collapse prior duplicates
                            cleaned_all = _clean_events_df(merged_all)
                            if detect_mode == "both" and not cleaned_all.empty:
                                cleaned_all = cleaned_all.sort_values("date")
                                cleaned_all = cleaned_all.drop_duplicates(subset=["date", "persistence"], keep="first")
                            st.session_state["events_df"] = cleaned_all

                        # Segment breakpoints from merged classified, with consolidation across series when needed
                        def _merge_month_end_dates(
                            dates: list[pd.Timestamp],
                            min_gap_months: int = 1,
                        ) -> list[pd.Timestamp]:
                            if not dates:
                                return []
                            ds = sorted({pd.to_datetime(d).to_period("M").to_timestamp("M") for d in dates})
                            merged: list[pd.Timestamp] = []
                            for d in ds:
                                if not merged:
                                    merged.append(d)
                                else:
                                    prev = merged[-1]
                                    if (d.to_period("M").ordinal - prev.to_period("M").ordinal) >= min_gap_months:
                                        merged.append(d)
                            return merged

                        seg_effects = filter_breakpoints(
                            merged_classified,
                            effects=["Persistent"],
                            components=["rate", "mixed"],
                        )
                        try:
                            logger.info(
                                "Classifier filtered (Persistent & rate/mixed): %s",
                                [
                                    {
                                        "index": int(getattr(b, "index", -1)),
                                        "date": str(getattr(b, "date", "")),
                                        "component": str(getattr(b, "component", "")),
                                    }
                                    for b in seg_effects
                                ],
                            )
                        except Exception:
                            pass

                        if detect_mode == "both":
                            # Merge by month-end to avoid double-counting near-duplicate breaks across Total/Paid
                            merged_dates = _merge_month_end_dates([b.date for b in seg_effects], min_gap_months=1)
                            base_index = (
                                plot_df["Total"].dropna().index
                                if "Total" in plot_df.columns
                                else (plot_df["Paid"].dropna().index if "Paid" in plot_df.columns else plot_df.index)
                            )
                            seg_bkps = [base_index.get_loc(d) for d in merged_dates if d in base_index]
                            seg_bkps = sorted(set(seg_bkps))
                            st.session_state["detected_change_dates"] = merged_dates
                        else:
                            # Single-series path: detector already de-dupes within-series; just map indices→dates
                            seg_bkps = sorted({b.index for b in seg_effects})
                            s_idx = plot_df[target_col].dropna().index if target_col is not None else plot_df.index
                            st.session_state["detected_change_dates"] = [s_idx[i] for i in seg_bkps if i < len(s_idx)]

                        st.session_state["detected_breakpoints"] = seg_bkps
                        with suppress(Exception):
                            logger.info(
                                "Detection result: mode=%s, label=%s, seg_bkps=%s, change_dates=%s",
                                detect_mode,
                                ",".join(label_list) if label_list else str(target_col),
                                seg_bkps,
                                [
                                    str(pd.to_datetime(d).date())
                                    for d in st.session_state.get("detected_change_dates", [])
                                ],
                            )
                        # Save a human-readable label of what we detected on
                        if detect_mode == "both":
                            st.session_state["detected_target_label"] = "Total+Paid"
                        elif label_list:
                            st.session_state["detected_target_label"] = ",".join(label_list)
                        else:
                            st.session_state["detected_target_label"] = target_col
                        st.session_state["detected_target_col"] = target_col

                        _set_markers_from_events()
                        st.rerun()

    # The editable grid. Do *not* overwrite events_df here unless the grid actually changed.
    st.data_editor(
        st.session_state["events_df"],
        num_rows="dynamic",
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "type": st.column_config.SelectboxColumn("Type", options=EVENT_TYPE_OPTIONS, width="medium"),
            "persistence": st.column_config.SelectboxColumn(
                "Effect on ΔS",
                options=["No effect", "Transient", "Persistent"],
                width="medium",
                help=(
                    "Transient = one-month shock (ΔS), S stays higher.\n"
                    "Persistent = ongoing uplift (ΔS) from this month onward."
                ),
            ),
            "notes": st.column_config.TextColumn("Notes", width="large"),
            "cost": st.column_config.NumberColumn(
                "Cost ($)", step=10.0, min_value=0.0, format="%.2f", help="For Ad spend ROI calc"
            ),
        },
        width="stretch",
        key="events_editor",
        on_change=_on_events_editor_change,
    )

    # Quick-add (do NOT force month-end here)
    with st.expander("Quick add event", expanded=False):
        with st.form("quick_add_event_form", clear_on_submit=True):
            qa_date = st.date_input("Date (YYYY-MM-DD)")
            qa_type = st.selectbox("Type", EVENT_TYPE_OPTIONS)
            qa_persist = st.selectbox("Effect", ["No effect", "Transient", "Persistent"], index=1)
            qa_cost = st.number_input("Cost ($)", min_value=0.0, step=10.0, value=0.0, format="%.2f")
            qa_notes = st.text_input("Notes", value="")
            submitted = st.form_submit_button("Add to Events")

        if submitted and qa_date is not None:
            new_row = {
                "date": pd.to_datetime(qa_date).date(),  # keep the user's exact day
                "type": qa_type,
                "persistence": qa_persist,
                "notes": qa_notes,
                "cost": float(qa_cost or 0.0),
            }
            base = st.session_state.get("events_df", pd.DataFrame(columns=EVENTS_COLUMNS))
            merged = (
                pd.DataFrame([new_row]) if base.empty else pd.concat([base, pd.DataFrame([new_row])], ignore_index=True)
            )
            st.session_state["events_df"] = _clean_events_df(merged)
            _set_markers_from_events()
            st.rerun()


def events_features_ui(plot_df: pd.DataFrame) -> None:
    with st.expander("Stage 2: Events & Features (monthly)", expanded=False):
        st.caption(
            "Encodes pulse/step features from Events and optional ad spend using the fixed ad response parameters."
        )
        # Optional ad spend file
        ad_file = st.file_uploader("Optional: Ad spend CSV (date, spend)", type=["csv", "xlsx", "xls"], key="ad_csv")

        # --- Protect the user-edited events from accidental in-place mutation downstream ---
        _ev_backup = st.session_state.get("events_df", pd.DataFrame(columns=EVENTS_COLUMNS))
        st.session_state["events_df"] = _ev_backup.copy(deep=True)
        try:
            lam_current = float(st.session_state.get("adstock_lambda", DEFAULT_ADSTOCK_LAMBDA))
            theta_current = float(st.session_state.get("ad_log_theta", DEFAULT_AD_LOG_THETA))
            covariates_df, features_df = build_events_features(
                plot_df, lam=lam_current, theta=theta_current, ad_file=ad_file
            )
        finally:
            # Always restore the user-owned table, even if the builder throws
            st.session_state["events_df"] = _ev_backup
        # ------------------------------------------------------------------------------

        st.session_state["adstock_lambda"] = lam_current
        st.session_state["ad_log_theta"] = theta_current
        st.session_state["covariates_df"] = covariates_df
        st.session_state["features_df"] = features_df
        st.markdown(f"Using λ={lam_current:0.2f}, θ={theta_current:0.0f} (adjustable from the sidebar).")
        st.markdown("**Outputs**: `events_df` (above), `covariates_df`, `features_df`.")
        st.dataframe(features_df.reset_index(), width="stretch")
        # Log Phase 1 readiness for Phase 2 handoff
        with suppress(Exception):
            _bkps = list(st.session_state.get("detected_breakpoints", []))
            _n_events = 0 if st.session_state.get("events_df") is None else len(st.session_state.get("events_df"))
            _n_ad_rows = len(covariates_df) if isinstance(covariates_df, pd.DataFrame) else 0
            logger.info(
                "Phase 1 outputs ready (breakpoints=%s, events=%d, ad_rows=%d)",
                _bkps,
                _n_events,
                _n_ad_rows,
            )
            logger.info("Use 'Download phase1.json' to save handoff to Phase 2.")
        # Phase 1 handoff: import portable artifact
        uploaded_phase1 = st.file_uploader("Load phase1.json", type=["json"], key="phase1_json")
        if uploaded_phase1 is not None:
            try:
                apply_phase_one_json(uploaded_phase1)
                st.success("Phase 1 artifact loaded. Rebuilding features and updating state…")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load phase1.json: {e}")


def adds_and_churn_ui(plot_df: pd.DataFrame) -> None:
    with st.expander("Stage 3: Adds & Churn (monthly)", expanded=False):
        st.caption("Totals-only path: derive gross adds from net deltas using churn rate heuristics.")

        c1, c2 = st.columns(2)
        with c1:
            churn_free_est = st.number_input(
                "Monthly churn rate (free)",
                min_value=0.0,
                max_value=1.0,
                value=float(_get_state("churn_free", 0.0)),
                step=0.001,
                format="%0.3f",
            )
        with c2:
            churn_paid_est = st.number_input(
                "Monthly churn rate (paid)",
                min_value=0.0,
                max_value=1.0,
                value=float(_get_state("churn_prem", 0.0)),
                step=0.001,
                format="%0.3f",
            )

        adds_df, churn_df = derive_adds_churn(plot_df, churn_free_est=churn_free_est, churn_paid_est=churn_paid_est)
        if not adds_df.empty or not churn_df.empty:
            st.session_state["adds_df"] = adds_df
            st.session_state["churn_df"] = churn_df
            st.markdown("**Outputs**: `adds_df`, `churn_df` (monthly, heuristics).")
            st.dataframe(adds_df.reset_index(), width="stretch")
            st.dataframe(churn_df.reset_index(), width="stretch")
            b1, b2 = st.columns(2)
            with b1:
                st.download_button(
                    "Download adds.csv",
                    data=adds_df.reset_index().to_csv(index=False).encode("utf-8"),
                    file_name="adds.csv",
                    mime="text/csv",
                )
            with b2:
                st.download_button(
                    "Download churn.csv",
                    data=churn_df.reset_index().to_csv(index=False).encode("utf-8"),
                    file_name="churn.csv",
                    mime="text/csv",
                )


def quick_fit_ui(plot_df: pd.DataFrame, breakpoints: list[int]) -> None:
    """Stage 4: fit piecewise logistic, wire in exogenous feature, and draw overlay/forecast."""
    st.subheader("Stage 4: Fit model")
    st.caption(
        "Fits on Total (preferred) or Free if Total is unavailable. "
        "Uses detected change points as segments. If built, uses the ad-response exogenous feature."
    )

    # Controls
    use_exog = False
    features_df = st.session_state.get("features_df")
    if isinstance(features_df, pd.DataFrame) and "ad_effect_log" in features_df.columns:
        use_exog = st.checkbox("Use ad-response feature in fit (γ_exog·ad_effect_log)", value=True)

    horizon_ahead = st.slider("Forecast months ahead", 0, 36, 12, 1)

    if st.button("Fit model and overlay"):
        try:
            # ----- choose series to fit -----
            fit_series_source = plot_df.get("Total") if "Total" in plot_df.columns else plot_df.get("Free")
            if fit_series_source is None or fit_series_source.empty:
                st.info("Need Total or Free series to fit.")
                return

            # ----- optional exogenous regressor -----
            extra_exog = None
            if use_exog and isinstance(features_df, pd.DataFrame):
                extra_exog = features_df["ad_effect_log"].astype(float)

            # ----- fit -----
            fit = fit_piecewise_logistic(
                total_series=fit_series_source,
                breakpoints=breakpoints,
                events_df=st.session_state.get("events_df"),
                extra_exog=extra_exog,
            )
            st.session_state["pwlog_fit"] = fit

            # ----- initialize sidebar override defaults if absent -----
            if "modelfit_K" not in st.session_state:
                st.session_state["modelfit_K"] = _safe_float(getattr(fit, "carrying_capacity", None), 0.0)
            if "modelfit_gamma_pulse" not in st.session_state:
                st.session_state["modelfit_gamma_pulse"] = _safe_float(getattr(fit, "gamma_pulse", None), 0.0)
            if "modelfit_gamma_step" not in st.session_state:
                st.session_state["modelfit_gamma_step"] = _safe_float(getattr(fit, "gamma_step", None), 0.0)
            if getattr(fit, "gamma_exog", None) is not None and "modelfit_gamma_exog" not in st.session_state:
                st.session_state["modelfit_gamma_exog"] = _safe_float(getattr(fit, "gamma_exog", None), 0.0)
            fit_r_list = coerce_list(getattr(fit, "segment_growth_rates", None))
            fit_intercepts = coerce_list(getattr(fit, "segment_intercepts", None))

            existing_r = coerce_list(st.session_state.get("modelfit_r"))
            if ("modelfit_r" not in st.session_state) or (len(existing_r) != len(fit_r_list)):
                st.session_state["modelfit_r"] = fit_r_list
                if fit_r_list:
                    last_fit_r = _safe_float(fit_r_list[-1], 0.0)
                    st.session_state["modelfit_r_last_value"] = last_fit_r
                    st.session_state["modelfit_r_last_default"] = last_fit_r
                    if "modelfit_r_last_input" in st.session_state:
                        logger.info(
                            "Dropping modelfit_r_last_input after refitting; last_fit_r=%s",
                            last_fit_r,
                        )
                    st.session_state.pop("modelfit_r_last_input", None)
            elif ("modelfit_r_last_value" not in st.session_state) and existing_r:
                last_existing_r = _safe_float(existing_r[-1], 0.0)
                st.session_state["modelfit_r_last_value"] = last_existing_r
                st.session_state["modelfit_r_last_default"] = last_existing_r
                if "modelfit_r_last_input" in st.session_state:
                    logger.info(
                        "Dropping modelfit_r_last_input while seeding from existing_r; "
                        "last_existing_r=%s",
                        last_existing_r,
                    )
                st.session_state.pop("modelfit_r_last_input", None)

            existing_intercepts = coerce_list(st.session_state.get("modelfit_intercepts"))
            if ("modelfit_intercepts" not in st.session_state) or (len(existing_intercepts) != len(fit_intercepts)):
                st.session_state["modelfit_intercepts"] = fit_intercepts

            # ----- read current overrides & recompute fitted line with them -----
            def _current_fit_params():
                fit_obj = st.session_state.get("pwlog_fit")
                k_src = st.session_state.get(
                    "modelfit_K",
                    getattr(fit_obj, "carrying_capacity", 0.0) if fit_obj is not None else 0.0,
                )
                k_val = _safe_float(k_src, 0.0)
                r_source = st.session_state.get(
                    "modelfit_r",
                    coerce_list(getattr(fit_obj, "segment_growth_rates", None)),
                )
                r_list = [_safe_float(val, 0.0) for val in coerce_list(r_source)]
                intercepts = coerce_list(
                    st.session_state.get(
                        "modelfit_intercepts",
                        coerce_list(getattr(fit_obj, "segment_intercepts", None)),
                    )
                )
                gp_src = st.session_state.get(
                    "modelfit_gamma_pulse",
                    getattr(fit_obj, "gamma_pulse", 0.0) if fit_obj is not None else 0.0,
                )
                gp_val = _safe_float(gp_src, 0.0)
                gs_src = st.session_state.get(
                    "modelfit_gamma_step",
                    getattr(fit_obj, "gamma_step", 0.0) if fit_obj is not None else 0.0,
                )
                gs_val = _safe_float(gs_src, 0.0)
                gx_raw = st.session_state.get("modelfit_gamma_exog", getattr(fit_obj, "gamma_exog", None))
                gx_val = None if gx_raw is None else _safe_float(gx_raw, 0.0)
                return k_val, r_list, intercepts, gp_val, gs_val, gx_val

            K_now, r_list_now, intercepts_now, gp_now, gs_now, gx_now = _current_fit_params()

            fitted_from_overrides = fitted_series_from_params(
                total_series=fit_series_source,
                breakpoints=breakpoints,
                carrying_capacity=float(K_now),
                segment_growth_rates=r_list_now,
                events_df=st.session_state.get("events_df"),
                extra_exog=(extra_exog if gx_now is not None else None),
                extra_exog_lag=(getattr(fit, "exog_lag", None) if gx_now is not None else None),
                gamma_pulse=float(gp_now),
                gamma_step=float(gs_now),
                gamma_exog=(float(gx_now) if gx_now is not None else None),
                segment_intercepts=intercepts_now,
            )

            # ----- overlay chart: Actual vs Fitted (overrides) -----
            overlay_df = pd.DataFrame(
                {"Actual": fit_series_source, "Fitted": fitted_from_overrides.reindex(fit_series_source.index)}
            )
            base_overlay = alt.Chart(overlay_df.reset_index().rename(columns={"index": "date"})).encode(
                x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y"))
            )
            actual_line = (
                base_overlay.transform_fold(["Actual"], as_=["Series", "Value"])
                .mark_line()
                .encode(y="Value:Q", color=alt.Color("Series:N", scale=alt.Scale(range=["#1f77b4"])))
            )
            fitted_line = (
                base_overlay.transform_fold(["Fitted"], as_=["Series", "Value"])
                .mark_line(strokeDash=[5, 3])
                .encode(y="Value:Q", color=alt.Color("Series:N", scale=alt.Scale(range=["#ff7f0e"])))
            )
            st.altair_chart(alt.layer(actual_line, fitted_line).properties(height=240), use_container_width=True)

            # ----- metrics -----
            c1, c2, c3 = st.columns(3)
            c1.metric("K (capacity)", f"{int(K_now):,}")
            last_r_metric = "—"
            if r_list_now:
                with suppress(Exception):
                    last_r_metric = f"{float(r_list_now[-1]):0.3f}"
            c2.metric("Last segment r", last_r_metric)
            c3.metric("R² on ΔS", f"{fit.r2_on_deltas:0.3f}")
            if getattr(fit, "gamma_exog", None) is not None:
                st.caption(f"Exogenous effect: γ_exog={float(gx_now):0.4f}")

            # ----- latex equation & stash for simulator tab -----
            eq = (
                r"\Delta S_t = r_{seg(t)}\, S_{t-1} \left(1 - \frac{S_{t-1}}{K}\right) "
                r"+ \gamma_{pulse}\,pulse_t + \gamma_{step}\,step_t"
            )
            if getattr(fit, "gamma_exog", None) is not None:
                eq += r" + \gamma_{exog}\,x_t"
            st.session_state["growth_equation_latex"] = eq
            with st.expander("Model equation and parameters", expanded=False):
                st.latex(eq)
                st.markdown("**Fitted parameters**")
                st.markdown(f"- **K (capacity)**: {float(K_now):,.0f}")
                last_r_text = "—"
                if r_list_now:
                    with suppress(Exception):
                        last_r_text = f"{float(r_list_now[-1]):0.3f}"
                st.markdown(f"- **Last segment growth rate (r)**: {last_r_text}")
                st.markdown(f"- **γ_pulse**: {gp_now:0.4f}")
                st.markdown(f"- **γ_step**: {gs_now:0.4f}")
                if gx_now is not None:
                    st.markdown(f"- **γ_exog**: {float(gx_now):0.4f}")

            # ----- optional forecast ahead -----
            if horizon_ahead > 0:
                last_val = float(fitted_from_overrides.iloc[-1])
                last_r = float(r_list_now[-1]) if r_list_now else 0.0
                fc = forecast_piecewise_logistic(
                    last_value=last_val,
                    months_ahead=horizon_ahead,
                    carrying_capacity=float(K_now),
                    segment_growth_rate=float(last_r),
                    gamma_step_level=float(gs_now),
                )
                fc_index = pd.date_range(
                    fitted_from_overrides.index[-1] + pd.offsets.MonthEnd(1),
                    periods=horizon_ahead,
                    freq="ME",
                )
                fc_df = pd.DataFrame({"Forecast": fc}, index=fc_index)
                merged = pd.concat([overlay_df, fc_df], axis=0)
                chart_fc = (
                    alt.Chart(merged.reset_index().rename(columns={"index": "date"}))
                    .transform_fold(["Actual", "Fitted", "Forecast"], as_=["Series", "Value"])
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y")),
                        y="Value:Q",
                        color="Series:N",
                    )
                    .properties(height=240)
                )
                st.altair_chart(chart_fc, use_container_width=True)

        except Exception as e:
            logger.exception("Model fit failed")
            st.exception(e)


def _current_fit_params():
    """Return current model parameters, preferring sidebar overrides when present.

    Returns (K, r_list, gamma_pulse, gamma_step, gamma_exog).
    """
    fit_obj = st.session_state.get("pwlog_fit")
    k_src = st.session_state.get(
        "modelfit_K",
        getattr(fit_obj, "carrying_capacity", 0.0) if fit_obj is not None else 0.0,
    )
    k_val = _safe_float(k_src, 0.0)
    r_source = st.session_state.get(
        "modelfit_r",
        coerce_list(getattr(fit_obj, "segment_growth_rates", None)),
    )
    r_list = [_safe_float(val, 0.0) for val in coerce_list(r_source)]
    gp_src = st.session_state.get(
        "modelfit_gamma_pulse",
        getattr(fit_obj, "gamma_pulse", 0.0) if fit_obj is not None else 0.0,
    )
    gp_val = _safe_float(gp_src, 0.0)
    gs_src = st.session_state.get(
        "modelfit_gamma_step",
        getattr(fit_obj, "gamma_step", 0.0) if fit_obj is not None else 0.0,
    )
    gs_val = _safe_float(gs_src, 0.0)
    gx_raw = st.session_state.get("modelfit_gamma_exog", getattr(fit_obj, "gamma_exog", None))
    gx_val = None if gx_raw is None else _safe_float(gx_raw, 0.0)
    return k_val, r_list, gp_val, gs_val, gx_val


def tail_view_ui(
    plot_df: pd.DataFrame, use_dual_axis: bool, show_total: bool, target_col: str | None, breakpoints: list[int]
) -> None:
    st.subheader("Stage 5: Diagnostics (delta view)")
    st.bar_chart(plot_df.diff().fillna(0))

    window_default = int(_get_state("est_window", 6))
    window = st.slider("Estimation window (last N months)", 3, 12, window_default, 1, key="est_window")
    st.caption("This window recomputes trailing medians for the estimates and the tail chart below.")

    st.subheader(f"Last {window} months (tail)")
    tail_df = plot_df.tail(window)

    series_title = "Series (Paid is dashed)" if (use_dual_axis and ("Paid" in tail_df.columns)) else "Series"
    base_chart = plot_series(tail_df, use_dual_axis=use_dual_axis, show_total=show_total, series_title=series_title)

    if target_col is not None and breakpoints:
        with suppress(Exception):
            full_s = plot_df[target_col].dropna()
            segs = compute_segment_slopes(full_s, breakpoints)
            tail_start, tail_end = tail_df.index[0], tail_df.index[-1]
            segs_t = [seg for seg in segs if (seg.end_date >= tail_start and seg.start_date <= tail_end)]
            fit_rows_t = []
            for seg in segs_t:
                xs = pd.date_range(max(seg.start_date, tail_start), min(seg.end_date, tail_end), freq="ME")
                start_val = float(full_s.loc[seg.start_date])
                fit_rows_t.extend({"date": d, "Fit": start_val + seg.slope_per_month * i} for i, d in enumerate(xs))
            if fit_rows_t:
                fit_df_t = pd.DataFrame(fit_rows_t)
                fit_t = (
                    alt.Chart(fit_df_t)
                    .mark_line(color="#7f8c8d")
                    .encode(
                        x=alt.X(
                            "date:T",
                            title="Date",
                            axis=alt.Axis(
                                labelExpr="timeFormat(datum.value, '%b %Y')",
                                labelAngle=0,
                                labelPadding=6,
                                titlePadding=10,
                            ),
                        ),
                        y="Fit:Q",
                    )
                )
                base_chart = alt.layer(base_chart, fit_t).resolve_scale(y="independent").properties(height=240)

    st.altair_chart(base_chart, use_container_width=True)


def metrics_and_apply_ui(all_series: pd.Series | None, paid_series: pd.Series | None, net_only: bool) -> None:
    estimates = _compute_estimates(all_series, paid_series, int(_get_state("est_window", 6)))

    cols = st.columns(3)
    if "start_free" in estimates:
        cols[0].metric("Starting free (latest)", f"{estimates['start_free']:,}")
    if "start_premium" in estimates:
        cols[1].metric("Starting premium (latest)", f"{estimates['start_premium']:,}")
    if "organic_growth" in estimates:
        cols[2].metric("Net free growth (monthly)", f"{estimates['organic_growth']*100:0.2f}%")

    cols2 = st.columns(3)
    if "conv_ongoing" in estimates:
        cols2[0].metric("Ongoing premium conversion (proxy)", f"{estimates['conv_ongoing']*100:0.3f}%")
    if not net_only:
        cols2[1].metric("Free churn", f"{float(estimates.get('churn_free', 0.0))*100:0.2f}%")
        cols2[2].metric("Premium churn", f"{float(estimates.get('churn_prem', 0.0))*100:0.2f}%")
    else:
        cols2[1].metric("Net growth (includes churn)", "—")
        cols2[2].metric(" ", " ")

    st.caption(
        "Notes: From totals alone we can compute net growth and a conversion proxy (when both series present). Churn and CAC need more detail."
    )

    # Offer the Phase 1 download near the final action
    st.download_button(
        "Download phase1.json",
        data=export_phase_one_json(),
        file_name="phase1.json",
        mime="application/json",
        help=(
            "Portable handoff from Phase 1 → Phase 2 " "(series, events, ad spend, breakpoints, knobs, fit parameters)"
        ),
    )

    if st.button("Apply estimates to Simulator"):
        for k, v in estimates.items():
            st.session_state[k] = v
        if net_only:
            st.session_state["churn_free"] = 0.0
            st.session_state["churn_prem"] = 0.0
        st.session_state["ad_stage1"] = 0.0
        st.session_state["ad_stage2"] = 0.0
        st.session_state["ad_const"] = 0.0
        st.session_state["spend_mode_index"] = 1
        st.session_state["conv_new"] = 0.0
        st.session_state["horizon_months"] = max(int(_get_state("horizon_months", 60)), 24)
        # Ensure the Simulator shows an equation even if model fit wasn't run
        if "growth_equation_latex" not in st.session_state:
            st.session_state["growth_equation_latex"] = (
                r"F_t = F_{t-1}(1 - c_f) + F_{t-1}\,g + \frac{AdSpend_t}{CAC} - conv_t\\"
                r"P_t = P_{t-1}(1 - c_p) + conv_t\\"
                r"conv_t = (new^{free}_t)\,p_{new} + F_{t-1}\,p_{ongoing},\\"
                r"\quad new^{free}_t = F_{t-1}\,g + \frac{AdSpend_t}{CAC}"
            )
        st.session_state["switch_to_sim"] = True
        st.success("Applied. Switching to Simulator…")
        st.rerun()


def number_input_state(label: str, *, key: str, default_value, **kwargs):
    kwargs["key"] = key

    def _is_number(value: Any) -> bool:
        return isinstance(value, numbers.Number)

    current_value = st.session_state.get(key, default_value)

    min_value = kwargs.get("min_value")
    if min_value is not None:
        candidates = [min_value, default_value, current_value]
        numeric_candidates = [val for val in candidates if _is_number(val)]
        if numeric_candidates:
            kwargs["min_value"] = min(numeric_candidates)

    max_value = kwargs.get("max_value")
    if max_value is not None:
        candidates = [max_value, default_value, current_value]
        numeric_candidates = [val for val in candidates if _is_number(val)]
        if numeric_candidates:
            kwargs["max_value"] = max(numeric_candidates)

    if key not in st.session_state:
        kwargs["value"] = default_value
    return st.number_input(label, **kwargs)


def slider_state(label: str, *, key: str, default_value, **kwargs):
    kwargs["key"] = key
    if key not in st.session_state:
        kwargs["value"] = default_value
    return st.slider(label, **kwargs)


def _render_include_checkboxes(has_fit: bool, fit_key: str, sim_key: str) -> tuple[bool, bool]:
    include_fit = st.checkbox("Include model fit", value=has_fit, key=fit_key)
    include_sim = st.checkbox("Include simulation results", value=False, key=sim_key)
    return include_fit, include_sim


def sidebar_inputs() -> SimulationInputs:
    st.sidebar.header("Assumptions")

    # Brief status: show fitted segment growth rates if available
    fit_side = st.session_state.get("pwlog_fit")
    r_overrides = coerce_list(st.session_state.get("modelfit_r"))
    if not r_overrides and fit_side is not None:
        r_overrides = coerce_list(getattr(fit_side, "segment_growth_rates", None))
    if r_overrides:
        last_r_caption = "—"
        with suppress(Exception):
            last_r_caption = f"{float(r_overrides[-1]):0.3f}"
        st.sidebar.caption(f"Last segment r: {last_r_caption}. Edit under Model fit parameters.")

    with st.sidebar.expander("Starting point", expanded=True):
        start_free = number_input_state(
            "Starting free subscribers",
            min_value=0,
            default_value=int(_get_state("start_free", 0)),
            step=10,
            key="start_free",
        )
        start_premium = number_input_state(
            "Starting premium subscribers",
            min_value=0,
            default_value=int(_get_state("start_premium", 0)),
            step=1,
            key="start_premium",
        )

    with st.sidebar.expander("Horizon", expanded=False):
        horizon = slider_state(
            "Months to simulate",
            min_value=12,
            max_value=120,
            default_value=int(_get_state("horizon_months", 60)),
            step=6,
            key="horizon_months",
        )

    with st.sidebar.expander("Growth & churn", expanded=True):
        organic_growth = number_input_state(
            "Organic monthly growth (free)",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("organic_growth", DEFAULT_GROWTH_RATE)),
            step=0.001,
            format="%0.3f",
            key="organic_growth",
        )
        churn_free = number_input_state(
            "Monthly churn (free)",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("churn_free", 0.0)),
            step=0.001,
            format="%0.3f",
            key="churn_free",
        )
        churn_prem = number_input_state(
            "Monthly churn (premium)",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("churn_prem", 0.0)),
            step=0.001,
            format="%0.3f",
            key="churn_prem",
        )

    with st.sidebar.expander("Conversions", expanded=True):
        conv_new = number_input_state(
            "New-subscriber premium conversion",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("conv_new", 0.0)),
            step=0.001,
            format="%0.3f",
            key="conv_new",
        )
        conv_ongoing = number_input_state(
            "Ongoing premium conversion of existing free",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("conv_ongoing", 0.0)),
            step=0.0001,
            format="%0.4f",
            key="conv_ongoing",
        )

    with st.sidebar.expander("Acquisition", expanded=True):
        spend_mode = st.selectbox(
            "Ad spend schedule",
            ["Two-stage (Years 1-2 / 3-5)", "Constant"],
            index=int(_get_state("spend_mode_index", 1)),
            key="spend_mode",
        )
        if spend_mode.startswith("Two-stage"):
            stage1 = number_input_state(
                "Monthly ad spend (years 1-2)",
                min_value=0.0,
                default_value=float(_get_state("ad_stage1", 0.0)),
                step=50.0,
                key="ad_stage1",
            )
            stage2 = number_input_state(
                "Monthly ad spend (years 3-5)",
                min_value=0.0,
                default_value=float(_get_state("ad_stage2", 0.0)),
                step=50.0,
                key="ad_stage2",
            )
            stage1 = float(stage1 or 0.0)
            stage2 = float(stage2 or 0.0)
            ad_schedule = AdSpendSchedule.two_stage(stage1, stage2)
            st.session_state["spend_mode_index"] = 0
        else:
            const_spend = number_input_state(
                "Monthly ad spend (constant)",
                min_value=0.0,
                default_value=float(_get_state("ad_const", 0.0)),
                step=50.0,
                key="ad_const",
            )
            const_spend = float(const_spend or 0.0)
            ad_schedule = AdSpendSchedule.constant(const_spend)
            st.session_state["spend_mode_index"] = 1

        with st.sidebar.expander("Ad spend preview", expanded=False):
            horizon_idx = max(int(horizon) - 1, 0)
            candidates = [0, 11, 23, 35, 59, horizon_idx]
            preview_months = sorted({min(max(m, 0), horizon_idx) for m in candidates})
            preview_rows = {
                f"Month {m + 1}": format_currency(float(ad_schedule.get_spend_for_month(m))) for m in preview_months
            }
            st.write("Representative monthly ad spend:", preview_rows)

        cac = number_input_state(
            "Cost per new free subscriber (CAC)",
            min_value=0.01,
            default_value=float(_get_state("cac", 2.0)),
            step=0.1,
            key="cac",
        )
        ad_manager_fee = number_input_state(
            "Ad manager monthly fee",
            min_value=0.0,
            default_value=float(_get_state("ad_manager_fee", 0.0)),
            step=50.0,
            key="ad_manager_fee",
        )

    with st.sidebar.expander("Pricing & fees", expanded=True):
        price_monthly = number_input_state(
            "Premium monthly price (gross)",
            min_value=0.0,
            default_value=float(_get_state("price_monthly", 10.0)),
            step=1.0,
            key="price_monthly",
        )
        price_annual = number_input_state(
            "Premium annual price (gross)",
            min_value=0.0,
            default_value=float(_get_state("price_annual", 70.0)),
            step=5.0,
            key="price_annual",
        )
        substack_pct = number_input_state(
            "Substack fee %",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("substack_pct", 0.10)),
            step=0.01,
            format="%0.2f",
            key="substack_pct",
        )
        stripe_pct = number_input_state(
            "Stripe % (billing + card)",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("stripe_pct", 0.036)),
            step=0.001,
            format="%0.3f",
            key="stripe_pct",
        )
        stripe_flat = number_input_state(
            "Stripe flat per transaction",
            min_value=0.0,
            default_value=float(_get_state("stripe_flat", 0.30)),
            step=0.05,
            key="stripe_flat",
        )
        annual_share = slider_state(
            "Share of premium on annual plans",
            min_value=0.0,
            max_value=1.0,
            default_value=float(_get_state("annual_share", 0.0)),
            step=0.05,
            key="annual_share",
        )

    # Ad response feature parameters (used in Stage 2 feature building)
    with st.sidebar.expander("Ad response (features)", expanded=False):
        lam_sb = slider_state(
            "Adstock lambda (carryover)",
            min_value=0.0,
            max_value=0.99,
            default_value=float(_get_state("adstock_lambda", DEFAULT_ADSTOCK_LAMBDA)),
            step=0.01,
            key="adstock_lambda",
        )
        theta_sb = number_input_state(
            "Log transform theta",
            min_value=1.0,
            default_value=float(_get_state("ad_log_theta", DEFAULT_AD_LOG_THETA)),
            step=50.0,
            key="ad_log_theta",
        )

    # Model fit parameters (read from fit; allow manual override for what-if scenarios)
    with st.sidebar.expander("Model fit parameters", expanded=True):
        fit = st.session_state.get("pwlog_fit")
        if fit is not None:
            logger.info(
                "Rendering model fit parameters: carrying_capacity=%s, gamma_pulse=%s, gamma_step=%s, gamma_exog=%s",
                getattr(fit, "carrying_capacity", None),
                getattr(fit, "gamma_pulse", None),
                getattr(fit, "gamma_step", None),
                getattr(fit, "gamma_exog", None),
            )
        if fit is None:
            st.caption(
                "No model fit available yet. Run Model fit on the Estimators tab or edit r below for simulations."
            )
        else:
            try:
                k_val = number_input_state(
                    "K (carrying capacity)",
                    min_value=0.0,
                    default_value=_safe_float(getattr(fit, "carrying_capacity", None), 0.0),
                    step=100.0,
                    key="modelfit_K",
                )
                gp = number_input_state(
                    "gamma_pulse",
                    min_value=-10.0,
                    max_value=10.0,
                    default_value=_safe_float(getattr(fit, "gamma_pulse", None), 0.0),
                    step=0.001,
                    key="modelfit_gamma_pulse",
                )
                gs = number_input_state(
                    "gamma_step",
                    min_value=-10.0,
                    max_value=10.0,
                    default_value=_safe_float(getattr(fit, "gamma_step", None), 0.0),
                    step=0.001,
                    key="modelfit_gamma_step",
                )
                gx0 = getattr(fit, "gamma_exog", None)
                if gx0 is not None:
                    gx = number_input_state(
                        "gamma_exog (log ad)",
                        min_value=-10.0,
                        max_value=10.0,
                        default_value=_safe_float(gx0, 0.0),
                        step=0.001,
                        key="modelfit_gamma_exog",
                    )

                # Segment growth rates r_j
                base_r_list = coerce_list(st.session_state.get("modelfit_r"))
                logger.info("Initial base_r_list from session state: %s", base_r_list)
                if not base_r_list:
                    base_r_list = coerce_list(getattr(fit, "segment_growth_rates", None))
                    logger.info("Fallback base_r_list from fit: %s", base_r_list)
            except Exception:
                logger.exception("Model fit parameters available, but could not render editor. fit=%s", fit)
                st.caption("Model fit parameters available, but could not render editor.")
                base_r_list = []
        if fit is None:
            base_r_list = coerce_list(st.session_state.get("modelfit_r"))
        if not base_r_list:
            base_r_list = [_safe_float(_get_state("organic_growth", DEFAULT_GROWTH_RATE), DEFAULT_GROWTH_RATE)]

        organic_default = _safe_float(_get_state("organic_growth", DEFAULT_GROWTH_RATE), DEFAULT_GROWTH_RATE)
        base_r_list = [_safe_float(val, organic_default) for val in coerce_list(base_r_list)]

        # Remove legacy per-segment widget keys to avoid stale values lingering in state
        preserved_r_keys = {
            "modelfit_r_last_input",
            "modelfit_r_last_value",
            "modelfit_r_last_default",
        }
        for key in list(st.session_state.keys()):
            if key.startswith("modelfit_r_") and key not in preserved_r_keys:
                del st.session_state[key]

        last_segment_default = float(base_r_list[-1] if base_r_list else organic_default)
        last_default = st.session_state.get("modelfit_r_last_default")
        if last_default is None or not math.isclose(
            last_default,
            last_segment_default,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            if last_default is None:
                logger.info(
                    "Resetting last segment input because no stored default exists; "
                    "computed_default=%s",
                    last_segment_default,
                )
            else:
                logger.info(
                    "Resetting last segment input due to default drift; "
                    "stored_default=%s, computed_default=%s",
                    last_default,
                    last_segment_default,
                )
            st.session_state["modelfit_r_last_default"] = last_segment_default
            st.session_state.pop("modelfit_r_last_input", None)
        last_r_val = number_input_state(
            "Last segment growth rate (r)",
            min_value=-10.0,
            max_value=10.0,
            default_value=last_segment_default,
            step=0.001,
            key="modelfit_r_last_input",
        )

        if base_r_list:
            r_over = list(base_r_list)
            r_over[-1] = float(last_r_val)
        else:
            r_over = [float(last_r_val)]

        # Persist aggregate list (non-widget key) for convenience
        st.session_state["modelfit_r"] = r_over
        if r_over:
            last_value = float(r_over[-1])
            st.session_state["modelfit_r_last_value"] = last_value
            st.session_state["modelfit_r_last_default"] = last_value

    # Map model-fit overrides into simulator: use last segment r as organic growth if available
    _k_now, _r_now, _gp_now, _gs_now, _gx_now = _current_fit_params()
    carrying_capacity = None
    try:
        k_float = float(_k_now)
    except (TypeError, ValueError):
        k_float = 0.0
    if k_float > 0:
        carrying_capacity = k_float
    organic_from_fit = float(_r_now[-1]) if (_r_now and len(_r_now) > 0) else float(organic_growth)

    return SimulationInputs(
        starting_free_subscribers=start_free,
        starting_premium_subscribers=start_premium,
        carrying_capacity=carrying_capacity,
        horizon_months=horizon,
        organic_monthly_growth_rate=organic_from_fit,
        monthly_churn_rate_free=float(churn_free),
        monthly_churn_rate_premium=float(churn_prem),
        new_subscriber_premium_conv_rate=float(conv_new),
        ongoing_premium_conv_rate=float(conv_ongoing),
        cost_per_new_free_subscriber=float(cac),
        ad_spend_schedule=ad_schedule,
        ad_manager_monthly_fee=float(ad_manager_fee),
        premium_monthly_price_gross=float(price_monthly),
        premium_annual_price_gross=float(price_annual),
        substack_fee_pct=float(substack_pct),
        stripe_fee_pct=float(stripe_pct),
        stripe_flat_fee=float(stripe_flat),
        annual_share=float(annual_share),
    )


def render_kpis(df: pd.DataFrame) -> None:
    last = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ending free", f"{int(last.free_subscribers):,}")
    col2.metric("Ending premium", f"{int(last.premium_subscribers):,}")
    col3.metric("Net MRR", format_currency(last.mrr_net))
    col4.metric("Cumulative profit", format_currency(last.cumulative_net_profit))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Cumulative ad spend", format_currency(last.cumulative_ad_spend))
    roas = 0.0 if last.cumulative_ad_spend == 0 else (df.net_revenue.sum() / df.ad_spend.sum())
    col6.metric("ROAS (net revenue / ad spend)", f"{roas:0.2f}x")
    avg_cac = float("nan") if df.new_free_paid.sum() == 0 else df.ad_spend.sum() / df.new_free_paid.sum()
    col7.metric("Blended CAC (paid only)", format_currency(avg_cac))
    payback_month = next((i + 1 for i, c in enumerate(df.cumulative_net_profit) if c > 0), math.nan)
    col8.metric("Payback month (cumulative)", "—" if math.isnan(payback_month) else str(int(payback_month)))


def render_charts(df: pd.DataFrame) -> None:
    st.subheader("Subscribers over time")
    st.line_chart(
        df[["free_subscribers", "premium_subscribers"]].rename(
            columns={
                "free_subscribers": "Free",
                "premium_subscribers": "Premium",
            }
        )
    )

    st.subheader("Revenue and profit")
    st.area_chart(
        df[["mrr_net", "net_revenue", "profit"]].rename(
            columns={
                "mrr_net": "Net MRR",
                "net_revenue": "Net revenue (monthly)",
                "profit": "Profit (monthly)",
            }
        )
    )

    st.subheader("Spend vs revenue")
    st.line_chart(
        df[["ad_spend", "ad_manager_fee", "net_revenue"]].rename(
            columns={
                "ad_spend": "Ad spend",
                "ad_manager_fee": "Ad manager fee",
                "net_revenue": "Net revenue",
            }
        )
    )


def render_estimators() -> None:
    st.subheader("Quick estimators from your Substack stats")

    with st.expander("Organic growth and churn", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            f_start = st.number_input("Free subs at start", min_value=0, value=1000, step=10)
            f_end = st.number_input("Free subs at end", min_value=0, value=1200, step=10)
            months = st.number_input("Months in period", min_value=1, value=3, step=1)
        with c2:
            paid_new_total = st.number_input("New free from ads in period", min_value=0, value=200, step=10)
            free_churn_total = st.number_input(
                "Free churn (unsubs + cleaning) in period", min_value=0, value=50, step=10
            )

        avg_free = max((f_start + f_end) / 2.0, 1.0)
        organic_new_total_no_churn = (f_end - f_start) - paid_new_total
        organic_rate_no_churn = organic_new_total_no_churn / max(months * avg_free, 1.0)

        organic_new_total_with_churn = (f_end - f_start) - paid_new_total + free_churn_total
        organic_rate_with_churn = organic_new_total_with_churn / max(months * avg_free, 1.0)

        churn_rate_est = free_churn_total / max(months * avg_free, 1.0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Organic monthly growth (w/ churn)", f"{organic_rate_with_churn*100:0.2f}%")
        m2.metric("Organic monthly growth (simple)", f"{organic_rate_no_churn*100:0.2f}%")
        m3.metric("Monthly churn (free)", f"{churn_rate_est*100:0.2f}%")

        st.caption(
            "Tip: Use Subscribers over time + exports. Count paid-attributed signups to estimate 'new free from ads'."
        )

    with st.expander("Event evaluation (pre/post trend)", expanded=False):
        st.caption("Add an event date to compare the slope before and after for Total or Free.")
        target = st.selectbox("Series", ["Total", "Free", "Paid"], index=0)
        date_str = st.text_input("Event date (YYYY-MM-DD)", value="")
        if st.session_state.get("sim_df") is not None:
            # Use imported series when available
            series_map = {}
            with suppress(Exception):
                series_map |= {
                    "Total": st.session_state.get("import_total"),
                    "Paid": st.session_state.get("import_paid"),
                }
                if series_map.get("Total") is not None and series_map.get("Paid") is not None:
                    series_map["Free"] = series_map["Total"] - series_map["Paid"]
            if (s := series_map.get(target)) is not None and date_str:
                try:
                    dt = pd.to_datetime(date_str)
                    pre, post = slope_around(s, dt, window=6)
                    st.metric("Pre slope (per month)", f"{pre:0.2f}")
                    st.metric("Post slope (per month)", f"{post:0.2f}")
                except Exception:
                    st.info("Provide a valid date inside your imported series range.")

    with st.expander("Event ROI (rough)", expanded=False):
        st.caption("For Ad spend events with a cost, compare pre/post slope and estimate incremental subs.")
        ev = st.session_state.get("events_df")
        total_series = st.session_state.get("import_total")
        if ev is not None and total_series is not None:
            ev2 = ev if ev.empty else ev.dropna(subset=["date"])
            if ev2 is not None and not ev2.empty:
                ev2 = ev2.copy()
                # Coerce invalid dates to NaT then drop them
                ev2["date"] = pd.to_datetime(ev2["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
                ev2 = ev2.dropna(subset=["date"])  # keep only rows with valid dates
                rows = []
                for _, r in ev2.iterrows():
                    d = r["date"]
                    # Skip rows where analysis cannot be computed
                    try:
                        pre, post = slope_around(total_series, d, window=6)
                        delta = post - pre
                    except Exception:
                        continue

                    # Safely coerce cost to a numeric value, defaulting to 0.0 on NaN/NaT/invalid
                    raw_cost = r.get("cost", 0.0)
                    cost_num = pd.to_numeric(raw_cost, errors="coerce")
                    cost = 0.0 if pd.isna(cost_num) else float(cost_num)

                    rows.append({"date": d, "type": r.get("type", ""), "slope_delta": delta, "cost": cost})
                if rows:
                    out = pd.DataFrame(rows)
                    st.dataframe(out, width="stretch")

    with st.expander("Acquisition cost (CAC)", expanded=True):
        spend = st.number_input("Ad spend in period", min_value=0.0, value=3000.0, step=50.0)
        paid_new = st.number_input("New free subscribers from ads in period", min_value=0, value=150, step=10)
        cac = float("nan") if paid_new == 0 else spend / paid_new
        st.metric("CAC (cost per new free subscriber)", format_currency(cac if cac == cac else 0.0))
        st.caption("Tip: From ad manager or Substack 'Where subscribers came from' tagged as paid.")

    with st.expander("Premium conversions", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            new_prem_from_new = st.number_input(
                "New premium from brand-new free (first month)", min_value=0, value=10, step=1
            )
            new_free_this_month = st.number_input("New free this month (all sources)", min_value=0, value=500, step=10)
            conv_new = 0.0 if new_free_this_month == 0 else new_prem_from_new / new_free_this_month
            st.metric("New-subscriber premium conversion", f"{conv_new*100:0.2f}%")
        with c2:
            upgrades_existing = st.number_input(
                "Premium upgrades from existing free in period", min_value=0, value=15, step=1
            )
            avg_free_base = st.number_input("Average free base in period", min_value=1, value=1200, step=10)
            months2 = st.number_input("Months measured", min_value=1, value=3, step=1, key="months2")
            conv_ongoing = upgrades_existing / max(months2 * avg_free_base, 1.0)
            st.metric("Ongoing premium conversion (monthly)", f"{conv_ongoing*100:0.3f}%")

    with st.expander("Net revenue per premium (sanity check)", expanded=False):
        price_m = st.number_input("Monthly gross price", min_value=0.0, value=10.0, step=1.0)
        sub_pct = st.number_input("Substack %", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%0.2f")
        stripe_pct = st.number_input("Stripe %", min_value=0.0, max_value=1.0, value=0.036, step=0.001, format="%0.3f")
        stripe_flat = st.number_input("Stripe flat", min_value=0.0, value=0.30, step=0.05)
        net = price_m * (1 - sub_pct - stripe_pct) - stripe_flat
        st.metric("Net monthly revenue per premium", format_currency(max(net, 0)))


def render_help() -> None:
    st.subheader("How the Substack Ads ROI Simulator works")
    st.caption("Walk through each phase to see how data flows from import to simulation.")

    total_phases = len(STAGE_PHASES)
    for idx, phase in enumerate(STAGE_PHASES):
        st.markdown(f"### {phase['title']}")
        summary = phase.get("summary")
        if summary:
            st.markdown(summary)

        for stage in phase.get("stages", []):
            st.markdown(f"**{stage['label']}**")
            details = stage.get("details", [])
            if details:
                st.markdown("\n".join(f"- {item}" for item in details))
            st.write("")

        if idx < total_phases - 1:
            st.markdown("---")

    st.markdown("---")
    st.subheader("Growth equation")
    equation = st.session_state.get("growth_equation_latex")
    if equation:
        st.latex(equation)
        st.caption("Captured from the most recent fit on the Data Import tab.")
    else:
        st.latex(DEFAULT_SIMULATOR_EQUATION_LATEX)
        st.caption("Default deterministic cohort equations used by the Simulator sidebar assumptions.")

    st.markdown("---")
    st.subheader("How to map Substack stats to this simulator")
    st.markdown(
        """
**Key Simulator inputs and where to source them**

| Simulator input | Where to pull the numbers | How to translate your data |
| --- | --- | --- |
| Organic monthly growth (free) | **Audience → Subscribers → Over time** (or export). | Organic new = total new free − new free from ads. Monthly rate ≈ organic new ÷ (months × average free base). |
| Monthly churn (free & premium) | **Audience → Subscribers → Unsubscribes** plus list cleaning totals. | Monthly churn ≈ total churned ÷ (months × average cohort size). |
| CAC (cost per new free) | Your ad manager or Substack source tags marked as paid. | CAC = ad spend ÷ new free from ads in the same period. |
| New-subscriber premium conversion | Recent month of **Revenue → Subscriptions** broken out by signup source. | New premium from first-month signups ÷ number of new free that month. |
| Ongoing premium conversion | Premium upgrades not tied to first-month signups. | Monthly conversion ≈ upgrades ÷ (months × average free). |
| Pricing & fees | Subscription settings, Substack/Stripe statements. | Defaults: Substack 10%, Stripe 3.6% + $0.30. Adjust to match your setup. |
| Ad spend schedule | Ad platform pacing or budgeting docs. | Two-stage schedule covers months 0–23 vs 24–59; constant spend is a single monthly number. |

Use the Estimators tab to turn exports into these inputs, then paste the results into the Simulator sidebar.
        """
    )


def render_save_load() -> None:
    st.subheader("Save / Load session")
    st.caption("Download a portable bundle to save your work, or upload to restore it later.")

    c1, c2 = st.columns(2)
    with c1:
        has_fit = st.session_state.get("pwlog_fit") is not None
        include_fit = st.checkbox("Include model fit", value=has_fit)
        include_sim = st.checkbox("Include simulation results", value=False)
        bundle = collect_session_bundle(include_fit, include_sim)
        st.download_button(
            "Download session bundle (.zip)",
            data=bundle,
            file_name="substack_session.zip",
            mime="application/zip",
        )
    with c2:
        uploaded = st.file_uploader("Upload session bundle (.zip)", type=["zip"], key="session_bundle")
        if uploaded is not None:
            try:
                apply_session_bundle(uploaded)
                st.success("Session restored. Switching to Simulator…")
                st.session_state["switch_to_sim"] = True
                st.session_state["markers_source"] = (
                    "events"
                    if isinstance(st.session_state.get("events_df"), pd.DataFrame)
                    and not st.session_state["events_df"].empty
                    else "detect"
                )
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load bundle: {e}")


def _compute_estimates(all_series: pd.Series | None, paid_series: pd.Series | None, window_months: int = 6) -> dict:
    return compute_estimates(all_series, paid_series, window_months)


@dataclass
class ImportContext:
    all_series: pd.Series | None
    paid_series: pd.Series | None
    plot_df: pd.DataFrame
    net_only: bool


def _to_monthly_last(s: pd.Series | None) -> pd.Series | None:
    if s is None or s.empty:
        return s
    s2 = pd.to_datetime(pd.Index(s.index), errors="coerce")
    s = pd.Series(pd.to_numeric(s.values, errors="coerce"), index=s2).dropna()
    return s.resample("ME").last()


def _build_plot_df(all_series: pd.Series | None, paid_series: pd.Series | None) -> pd.DataFrame:
    plot_df = pd.DataFrame()
    if all_series is not None and not all_series.empty:
        plot_df["Total"] = all_series
    if paid_series is not None and not paid_series.empty:
        plot_df["Paid"] = paid_series
    if {"Total", "Paid"}.issubset(plot_df.columns):
        plot_df["Free"] = pd.to_numeric(plot_df["Total"], errors="coerce") - pd.to_numeric(
            plot_df["Paid"], errors="coerce"
        )
    return plot_df


def _safe_select_columns(head: pd.DataFrame, key_prefix: str) -> tuple[int | None, int | None]:
    ncols = head.shape[1]
    if ncols < 2:
        st.error(f"{key_prefix.capitalize()} file needs at least 2 columns (date, count).")
        return None, None
    date_sel = st.selectbox(
        f"{key_prefix.capitalize()}: date column (index)",
        list(range(ncols)),
        index=0,
        key=f"{key_prefix}_date_sel",
    )
    count_sel = st.selectbox(
        f"{key_prefix.capitalize()}: count column (index)",
        list(range(ncols)),
        index=min(1, ncols - 1),
        key=f"{key_prefix}_count_sel",
    )
    return date_sel, count_sel


def _ui_upload_two_files() -> tuple[Any | None, bool, int | None, int | None, Any | None, bool, int | None, int | None]:
    logger.info("Uploading two files")
    c_all, c_paid = st.columns(2)
    with c_all:
        all_file, all_has_header, all_date_sel, all_count_sel = upload_panel(
            "All subscribers file (CSV/XLSX, often downloaded as `[blogname]_emails_[date].csv`)",
            help_hint="Pick the time series of all subscribers over time.",
            key_prefix="all",
            default_header=False,
        )

    with c_paid:
        paid_file, paid_has_header, paid_date_sel, paid_count_sel = upload_panel(
            "Paid subscribers file (CSV/XLSX, often downloaded as `[blogname]_subscribers_[date].csv`)",
            help_hint="Pick the time series of paid subscribers over time.",
            key_prefix="paid",
            default_header=False,
        )

    return (
        all_file,
        all_has_header,
        all_date_sel,
        all_count_sel,
        paid_file,
        paid_has_header,
        paid_date_sel,
        paid_count_sel,
    )


def _parse_and_normalize_series(
    all_file,
    all_has_header,
    all_date_sel,
    all_count_sel,
    paid_file,
    paid_has_header,
    paid_date_sel,
    paid_count_sel,
) -> ImportContext:
    all_series = (
        read_series(all_file, all_has_header, all_date_sel, all_count_sel)
        if all_file is not None and all_date_sel is not None and all_count_sel is not None
        else None
    )
    paid_series = (
        read_series(paid_file, paid_has_header, paid_date_sel, paid_count_sel)
        if paid_file is not None and paid_date_sel is not None and paid_count_sel is not None
        else None
    )

    # Normalize to monthly once
    all_series_m = _to_monthly_last(all_series)
    paid_series_m = _to_monthly_last(paid_series)

    plot_df = _build_plot_df(all_series_m, paid_series_m)

    # Persist minimal state for other tabs
    if all_series_m is not None:
        st.session_state["import_total"] = all_series_m
    if paid_series_m is not None:
        st.session_state["import_paid"] = paid_series_m

    net_only = st.checkbox("Use net-only growth (set churn to 0)", value=True)
    return ImportContext(all_series_m, paid_series_m, plot_df, net_only)


def _ui_series_chart(plot_df: pd.DataFrame) -> tuple[bool, bool]:

    if plot_df.empty:
        return False, False
    use_dual_axis = st.checkbox(
        "Use separate right axis for Paid",
        value=True,
        help="Plots Total/Free on left axis and Paid on right axis for readability.",
    )
    show_total = st.checkbox("Show Total line", value=("Paid" not in plot_df.columns))
    series_title = "Series (Paid is dashed)" if (use_dual_axis and "Paid" in plot_df.columns) else "Series"
    base = plot_series(plot_df, use_dual_axis=use_dual_axis, show_total=show_total, series_title=series_title)
    event_rules = _event_rules_from_events() if st.session_state.get("markers_source", "events") == "events" else None
    chart = alt.layer(base, event_rules) if event_rules is not None else base
    st.altair_chart(chart, use_container_width=True)
    return use_dual_axis, show_total


def _stage2_events_and_detection(plot_df: pd.DataFrame) -> tuple[list[int], str | None]:
    target_col = "Total" if "Total" in plot_df.columns else ("Free" if "Free" in plot_df.columns else None)
    events_editor(plot_df, target_col)
    # Map Change events to breakpoints; show detected as reference only
    detected = trend_detection_ui(plot_df, target_col)
    change_dates = _events_change_dates()
    idx = plot_df[target_col].dropna().index if target_col is not None else plot_df.index
    # Only use breakpoints corresponding to rate/mixed from classification if available
    bkps_from_events = _dates_to_breakpoint_indices(change_dates, idx)
    # Prefer classifier-detected indices (stored in session_state["detected_breakpoints"]) over event-derived
    bkps_from_classifier = list(st.session_state.get("detected_breakpoints", []))
    chosen = bkps_from_classifier or bkps_from_events or detected
    return chosen, target_col


def _stage5_tail(
    plot_df: pd.DataFrame, use_dual_axis: bool, show_total: bool, target_col: str | None, breakpoints: list[int]
) -> None:
    tail_view_ui(plot_df, use_dual_axis, show_total, target_col, breakpoints)


def render_data_import() -> None:
    """
    Stage 1–5: Import, preview, annotate, feature-build, quick-fit, diagnostics, and handoff.
    """
    st.subheader("Stage 1: Import Substack exports (time series)")
    st.caption(
        "Upload two files: All subscribers over time, and Paid subscribers over time. "
        "We normalize everything to end-of-month (monthly). No headers by default: first column is date, second is count."
    )

    logger.info("Stage 1: entering Data Import")

    # Quick save/load
    with st.expander("Save / Load (quick access)", expanded=False):
        has_fit_i = st.session_state.get("pwlog_fit") is not None
        include_fit_i, include_sim_i = _render_include_checkboxes(has_fit_i, "import_include_fit", "import_include_sim")
        bundle_i = collect_session_bundle(include_fit_i, include_sim_i)
        st.download_button(
            "Export my config (.zip)",
            data=bundle_i,
            file_name="substack_session.zip",
            mime="application/zip",
            key="import_export_btn",
        )
        uploaded_i = st.file_uploader("Restore session bundle (.zip)", type=["zip"], key="import_session_bundle")
        if uploaded_i is not None:
            try:
                apply_session_bundle(uploaded_i)
                st.success("Session restored. Switching to Simulator…")
                st.session_state["switch_to_sim"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load bundle: {e}")

    # Uploads
    (
        all_file,
        all_has_header,
        all_date_sel,
        all_count_sel,
        paid_file,
        paid_has_header,
        paid_date_sel,
        paid_count_sel,
    ) = _ui_upload_two_files()

    # Only proceed if at least one file present
    if all_file is None and paid_file is None:
        return

    try:
        ctx = _parse_and_normalize_series(
            all_file,
            all_has_header,
            all_date_sel,
            all_count_sel,
            paid_file,
            paid_has_header,
            paid_date_sel,
            paid_count_sel,
        )

        if ctx.plot_df.empty:
            st.info("No usable data found after parsing/normalization.")
            return

        st.subheader("Imported series")
        st.caption("Mode: Paid and unpaid" if "Paid" in ctx.plot_df.columns else "Mode: Unpaid only")

        # Stage 1: observations
        if not ctx.plot_df.empty:
            emit_observations(ctx.plot_df)

        # Chart
        use_dual_axis, show_total = _ui_series_chart(ctx.plot_df)

        # Stage 2: events + detection + features
        breakpoints, target_col = _stage2_events_and_detection(ctx.plot_df)
        events_features_ui(ctx.plot_df)

        # Stage 3: Adds & Churn
        adds_and_churn_ui(ctx.plot_df)

        # Stage 4: Fit
        ev_dates_log = _events_change_dates()
        logger.info(
            "Stage 4: inputs — breakpoints=%s; change_dates=%s; plot_df_head=%s",
            breakpoints,
            [str(pd.to_datetime(d).date()) for d in (ev_dates_log or [])],
            ctx.plot_df.head(5).to_dict(orient="records"),
        )
        with st.expander("Stage 4 inputs (data & breakpoints)", expanded=False):
            st.write("Breakpoints (indices from 'Change' events):", breakpoints)
            st.write("Change event dates:", _events_change_dates())
            st.dataframe(ctx.plot_df, width="stretch")
        quick_fit_ui(ctx.plot_df, breakpoints)

        # Stage 5: Diagnostics tail
        _stage5_tail(ctx.plot_df, use_dual_axis, show_total, target_col, breakpoints)

        # Summary metrics + apply to simulator
        metrics_and_apply_ui(ctx.all_series, ctx.paid_series, net_only=ctx.net_only)

    except Exception as e:
        st.error(f"Estimation failed: {e}")


render_brand_header()

# Tabs
with st.container():
    tab_import, tab_sim, tab_est, tab_save, tab_help = st.tabs(
        [
            "Data Import",
            "Simulator",
            "Estimators",
            "Save / Load",
            "Help",
        ]
    )

with tab_import:
    render_data_import()

with tab_sim:
    inputs = sidebar_inputs()
    result = simulate_growth(inputs)
    sim_df = result.monthly
    st.session_state["sim_df"] = sim_df
    st.subheader("Stage 7: Cohort & Finance Simulator")
    st.info(
        "Set sidebar assumptions (growth, churn, conversion, pricing, CAC, ad spend) and run the Simulator "
        "to project subscribers, revenue, ROAS, and payback. The Help tab documents the workflow and "
        "current growth equation."
    )
    with st.expander("Save / Load (quick access)", expanded=False):
        has_fit_q = st.session_state.get("pwlog_fit") is not None
        include_fit_q = st.checkbox("Include model fit", value=has_fit_q, key="quick_include_fit")
        include_sim_q = st.checkbox("Include simulation results", value=False, key="quick_include_sim")
        bundle_q = collect_session_bundle(include_fit_q, include_sim_q)
        st.download_button(
            "Export my config (.zip)",
            data=bundle_q,
            file_name="substack_session.zip",
            mime="application/zip",
            key="quick_export_btn",
        )
        uploaded_q = st.file_uploader("Restore session bundle (.zip)", type=["zip"], key="quick_session_bundle")
        if uploaded_q is not None:
            try:
                apply_session_bundle(uploaded_q)
                st.success("Session restored. Switching to Simulator…")
                st.session_state["switch_to_sim"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load bundle: {e}")
    render_kpis(sim_df)
    with st.expander("Monthly details", expanded=False):
        st.dataframe(sim_df, width="stretch")
    render_charts(sim_df)

with tab_est:
    render_estimators()

with tab_save:
    render_save_load()

with tab_help:
    render_help()

# If requested, auto-switch to the Simulator tab by simulating a click
if st.session_state.get("switch_to_sim"):
    components.html(
        """
        <script>
        const tabs = parent.document.querySelectorAll('button[role="tab"]');
        for (const t of tabs) {
            if (t.innerText.trim() === 'Simulator') { t.click(); break; }
        }
        </script>
        """,
        height=0,
    )
    st.session_state["switch_to_sim"] = False
