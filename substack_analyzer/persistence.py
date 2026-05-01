import io
import json
import math
import zipfile
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from substack_analyzer.types import EventRow, PhaseOneOutput

# Stabilize Streamlit session_state when running headless (pytest or plain Python)
try:
    from streamlit import runtime as _st_runtime

    if not _st_runtime.exists():
        import streamlit.runtime.state.session_state_proxy as _ssp
        from streamlit.runtime.state.safe_session_state import SafeSessionState as _SafeSS
        from streamlit.runtime.state.session_state import SessionState as _SS

        if not hasattr(st, "_sa_headless_state"):
            st._sa_headless_state = _SafeSS(_SS(), lambda: None)
        _ssp.get_session_state = lambda: st._sa_headless_state  # type: ignore[assignment]
except Exception:
    pass


def collect_session_bundle(include_fit: bool, include_sim: bool) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        meta = {
            "schema_version": 1,
            "app_name": "Substack Ads ROI Simulator",
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        zf.writestr("metadata.json", json.dumps(meta, indent=2))

        # state: only include scalar configuration keys (avoid DataFrames/Series)
        allowed_keys = [
            "start_free",
            "start_premium",
            "horizon_months",
            "organic_growth",
            "churn_free",
            "churn_prem",
            "downgrade_to_free",
            "carrying_capacity_free",
            "carrying_capacity_premium",
            "conv_new",
            "conv_ongoing",
            "cac",
            "cac_premium",
            "ad_split_free_pct",
            "ad_manager_fee",
            "price_monthly",
            "price_annual",
            "substack_pct",
            "stripe_pct",
            "stripe_flat",
            "annual_share",
            "total_spend_mode_index",
            "total_ad_stage1",
            "total_ad_stage2",
            "total_ad_const",
            "total_ad_one_time_amount",
            "total_ad_one_time_month",
            "adstock_lambda",
            "ad_log_theta",
            "est_window",
            "max_changes_detect",
        ]
        state: dict[str, object] = {}
        for k in allowed_keys:
            if k in st.session_state:
                v = st.session_state.get(k)
                if hasattr(v, "item"):
                    with suppress(Exception):
                        v = v.item()
                if isinstance(v, (int, float, str, bool)) or v is None:
                    state[k] = v
        zf.writestr("state.json", json.dumps(state, indent=2))

        # series
        total = st.session_state.get("import_total")
        if isinstance(total, pd.Series) and not total.empty:
            df_t = total.rename("count").to_frame()
            df_t.index.name = "date"
            zf.writestr("series_total.csv", df_t.to_csv(index=True))

        paid = st.session_state.get("import_paid")
        if isinstance(paid, pd.Series) and not paid.empty:
            df_p = paid.rename("count").to_frame()
            df_p.index.name = "date"
            zf.writestr("series_paid.csv", df_p.to_csv(index=True))

        # events
        ev = st.session_state.get("events_df")
        if isinstance(ev, pd.DataFrame) and not ev.empty:
            ev_out = ev.copy()
            with suppress(Exception):
                ev_out["date"] = pd.to_datetime(ev_out["date"]).dt.date.astype(str)
            zf.writestr("events.csv", ev_out.to_csv(index=False))

        # covariates
        covariates = st.session_state.get("covariates_df")
        if isinstance(covariates, pd.DataFrame) and not covariates.empty:
            cov_out = covariates.copy()
            cov_out = cov_out.reset_index().rename(columns={cov_out.index.name or "index": "date"})
            with suppress(Exception):
                cov_out["date"] = pd.to_datetime(cov_out["date"]).dt.date.astype(str)
            zf.writestr("covariates.csv", cov_out.to_csv(index=False))

        # features
        features = st.session_state.get("features_df")
        if isinstance(features, pd.DataFrame) and not features.empty:
            feat_out = features.copy()
            feat_out = feat_out.reset_index().rename(columns={feat_out.index.name or "index": "date"})
            with suppress(Exception):
                feat_out["date"] = pd.to_datetime(feat_out["date"]).dt.date.astype(str)
            zf.writestr("features.csv", feat_out.to_csv(index=False))

        # fit
        if include_fit and (fit := st.session_state.get("pwlog_fit")) is not None:
            with suppress(Exception):
                fit_dict = {
                    "carrying_capacity": float(getattr(fit, "carrying_capacity", 0.0)),
                    "segment_growth_rates": [float(x) for x in getattr(fit, "segment_growth_rates", [])],
                    "breakpoints": list(getattr(fit, "breakpoints", [])),
                    "gamma_pulse": float(getattr(fit, "gamma_pulse", 0.0)),
                    "gamma_step": float(getattr(fit, "gamma_step", 0.0)),
                    "r2_on_deltas": float(getattr(fit, "r2_on_deltas", 0.0)),
                }
                zf.writestr("fit.json", json.dumps(fit_dict, indent=2))
                fitted = getattr(fit, "fitted_series", None)
                if isinstance(fitted, pd.Series) and not fitted.empty:
                    df_f = fitted.rename("fitted").to_frame()
                    df_f.index.name = "date"
                    zf.writestr("fit_fitted_series.csv", df_f.to_csv(index=True))

        # simulation
        if include_sim and (sim := st.session_state.get("sim_df")) is not None:
            with suppress(Exception):
                zf.writestr("sim.csv", sim.to_csv(index=False))

    buf.seek(0)
    return buf.getvalue()


def apply_session_bundle(file_like) -> None:
    # In headless/tests, Streamlit may not maintain a persistent SessionState.
    # Create a stable SafeSessionState and monkeypatch the getter so subsequent
    # accesses within this process see the same backing state.
    try:
        from streamlit import runtime as _st_runtime

        if not _st_runtime.exists():
            import streamlit.runtime.state.session_state_proxy as _ssp
            from streamlit.runtime.state.safe_session_state import SafeSessionState as _SafeSS
            from streamlit.runtime.state.session_state import SessionState as _SS

            if not hasattr(st, "_sa_headless_state"):
                st._sa_headless_state = _SafeSS(_SS(), lambda: None)
            _ssp.get_session_state = lambda: st._sa_headless_state  # type: ignore[assignment]
    except Exception:
        pass

    with zipfile.ZipFile(file_like, mode="r") as zf:
        # Ensure keys exist up-front in headless/test environments
        if "import_total" not in st.session_state:
            st.session_state["import_total"] = pd.Series(dtype=float)
        if "import_paid" not in st.session_state:
            st.session_state["import_paid"] = pd.Series(dtype=float)
        # metadata
        with suppress(KeyError, Exception):
            meta = json.loads(zf.read("metadata.json"))
            if int(meta.get("schema_version", 0)) != 1:
                raise ValueError("Unsupported bundle version. Please update the app.")

        # state
        with suppress(KeyError, Exception):
            state = json.loads(zf.read("state.json"))
            if isinstance(state, dict):
                # Apply scalar config state directly to session state in headless/tests
                for k, v in state.items():
                    st.session_state[k] = v

        # series: total
        with suppress(KeyError, Exception):
            df_t = pd.read_csv(io.BytesIO(zf.read("series_total.csv")))
            if {"date", "count"}.issubset(df_t.columns):
                s_t = (
                    df_t.assign(date=lambda d: pd.to_datetime(d["date"]))
                    .dropna(subset=["date"])
                    .set_index("date")["count"]
                )
                s_t = pd.to_numeric(s_t, errors="coerce").dropna()
                if not s_t.empty:
                    s_t.index = s_t.index.to_period("M").to_timestamp("M")
                    st.session_state.update({"import_total": s_t.sort_index()})

        # series: paid
        with suppress(KeyError, Exception):
            df_p = pd.read_csv(io.BytesIO(zf.read("series_paid.csv")))
            if {"date", "count"}.issubset(df_p.columns):
                s_p = (
                    df_p.assign(date=lambda d: pd.to_datetime(d["date"]))
                    .dropna(subset=["date"])
                    .set_index("date")["count"]
                )
                s_p = pd.to_numeric(s_p, errors="coerce").dropna()
                if not s_p.empty:
                    s_p.index = s_p.index.to_period("M").to_timestamp("M")
                    st.session_state.update({"import_paid": s_p.sort_index()})

        # events
        with suppress(KeyError, Exception):
            ev = pd.read_csv(io.BytesIO(zf.read("events.csv")))
            if not ev.empty:
                with suppress(Exception):
                    ev["date"] = pd.to_datetime(ev["date"]).dt.date
                st.session_state.update({"events_df": ev})

        # covariates
        with suppress(KeyError, Exception):
            cov = pd.read_csv(io.BytesIO(zf.read("covariates.csv")))
            if {"date", "ad_spend"}.issubset(cov.columns):
                cov["date"] = pd.to_datetime(cov["date"]).dt.to_period("M").dt.to_timestamp("M")
                st.session_state.update({"covariates_df": cov.set_index("date").sort_index()})

        # features
        with suppress(KeyError, Exception):
            feat = pd.read_csv(io.BytesIO(zf.read("features.csv")))
            need = {"date", "pulse", "step", "adstock", "ad_effect_log"}
            if need.issubset(feat.columns):
                feat["date"] = pd.to_datetime(feat["date"]).dt.to_period("M").to_timestamp("M")
                st.session_state.update({"features_df": feat.set_index("date").sort_index()})

        # Ensure required keys exist even if the bundle lacks certain artifacts
        # This helps in headless/test environments where session_state may not persist
        if "import_total" not in st.session_state:
            st.session_state.update({"import_total": pd.Series(dtype=float)})
        if "import_paid" not in st.session_state:
            st.session_state.update({"import_paid": pd.Series(dtype=float)})


def _series_to_records(s: pd.Series | None) -> list[dict[str, Any]] | None:
    if s is None or not isinstance(s, pd.Series) or s.empty:
        return None
    s2 = s.dropna()
    if s2.empty:
        return None
    idx = s2.index.to_period("M").to_timestamp("M")
    df = pd.DataFrame({"date": idx, "count": s2.astype(float).values})
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df.to_dict(orient="records")


def _records_to_series(records: list[dict[str, Any]] | None) -> pd.Series | None:
    if not records:
        return None
    df = pd.DataFrame(records)
    if not {"date", "count"}.issubset(df.columns):
        return None
    df = df.assign(date=lambda d: pd.to_datetime(d["date"]))
    # Keep rows where both date and count are valid; ensure alignment after dropna
    counts = pd.to_numeric(df["count"], errors="coerce")
    df = df.loc[counts.notna()].dropna(subset=["date"]).copy()
    if df.empty:
        return None
    df = df.sort_values("date")
    idx = df["date"].dt.to_period("M").dt.to_timestamp("M")
    return pd.Series(counts.loc[df.index].astype(float).values, index=idx)


def _detect_display_to_code(value: str) -> str:
    """
    Normalize UI/display detect mode labels to canonical codes used in artifacts:
      - "Both (Total+Paid)" -> "both"
      - "Auto (Total→Free)" or "Auto" or "Default" -> "auto"
      - "Total" -> "total"
      - "Free" -> "free"
      - "Paid" -> "paid"
    If the input already looks like a code (e.g., 'both'), it is returned lower-cased.
    """
    s = str(value or "").strip()
    if not s:
        return "auto"
    low = s.lower()
    if low in {"auto", "default"} or low.startswith("auto"):
        return "auto"
    if low.startswith("both") or "total+paid" in low:
        return "both"
    if low.startswith("total"):
        return "total"
    if low.startswith("free"):
        return "free"
    if low.startswith("paid"):
        return "paid"
    # Fallback: return as lower-case to preserve unknown custom values
    return low


def _detect_code_to_display(code: str) -> str:
    """
    Map canonical codes to UI/display labels used by the app selectbox.
    Unknown codes are returned unchanged.
    """
    c = str(code or "").strip().lower()
    if c in {"", "auto", "default"}:
        return "Auto (Total\u2192Free)"
    if c == "both":
        return "Both (Total+Paid)"
    if c == "total":
        return "Total"
    if c == "free":
        return "Free"
    if c == "paid":
        return "Paid"
    return code


def export_phase_one_json() -> bytes:
    """Serialize Phase 1 outputs into a portable JSON artifact.

    Reads from session_state the monthly series, detected breakpoints/dates, events,
    ad spend (monthly), and feature knobs; returns JSON bytes.
    """
    total = st.session_state.get("import_total")
    paid = st.session_state.get("import_paid")
    events_df = st.session_state.get("events_df")
    cov_df = st.session_state.get("covariates_df")

    ev_rows: list[dict[str, Any]] = []
    if isinstance(events_df, pd.DataFrame) and not events_df.empty:
        e2 = events_df.copy()
        with suppress(Exception):
            e2["date"] = pd.to_datetime(e2["date"]).dt.date.astype(str)
        for _, r in e2.iterrows():
            ev_rows.append(
                EventRow(
                    date=str(r.get("date", "")),
                    type=str(r.get("type", "")),
                    persistence=str(r.get("persistence", "")),
                    notes=str(r.get("notes", "")),
                    cost=float(r.get("cost", 0.0) or 0.0),
                ).__dict__
            )

    ad_spend_records: list[dict[str, Any]] | None = None
    if isinstance(cov_df, pd.DataFrame) and ("ad_spend" in cov_df.columns) and not cov_df.empty:
        df = cov_df[["ad_spend"]].copy()
        df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        with suppress(Exception):
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        ad_spend_records = df.rename(columns={"ad_spend": "spend"}).to_dict(orient="records")

    payload: dict[str, Any] = asdict(
        PhaseOneOutput(
            total_series=_series_to_records(total),
            paid_series=_series_to_records(paid),
            breakpoints_indices=list(st.session_state.get("detected_breakpoints", []) or []),
            breakpoints_dates=[
                str(pd.to_datetime(d).date()) for d in (st.session_state.get("detected_change_dates", []) or [])
            ],
            events=[EventRow(**r) for r in ev_rows],
            ad_spend=ad_spend_records,
            adstock_lambda=float(st.session_state.get("adstock_lambda", 0.5)),
            ad_log_theta=float(st.session_state.get("ad_log_theta", 500.0)),
            # Export canonical code
            detect_mode=str(st.session_state.get("detect_on", "auto")).strip().lower(),
            detected_target_label=st.session_state.get("detected_target_label"),
            target_col_for_fit=st.session_state.get("detected_target_col"),
        )
    )

    # If a model fit is present, include its parameters for Phase 2 equation-based simulation
    fit = st.session_state.get("pwlog_fit")
    # Prefer live overrides from the UI when present; otherwise fall back to the fitted object

    def _rf(v: float | None, nd: int = 6) -> float | None:
        if v is None:
            return None
        return round(float(v), nd)

    k_override = st.session_state.get("modelfit_K")
    r_override = st.session_state.get("modelfit_r")
    gp_override = st.session_state.get("modelfit_gamma_pulse")
    gs_override = st.session_state.get("modelfit_gamma_step")
    gx_override = st.session_state.get("modelfit_gamma_exog")

    k_val = (
        float(k_override)
        if k_override is not None
        else (float(getattr(fit, "carrying_capacity", 0.0)) if fit is not None else None)
    )
    r_list = (
        list(r_override)
        if isinstance(r_override, (list, tuple))
        else (list(getattr(fit, "segment_growth_rates", [])) if fit is not None else None)
    )
    gp_val = (
        float(gp_override)
        if gp_override is not None
        else (float(getattr(fit, "gamma_pulse", 0.0)) if fit is not None else 0.0)
    )
    gs_val = (
        float(gs_override)
        if gs_override is not None
        else (float(getattr(fit, "gamma_step", 0.0)) if fit is not None else 0.0)
    )
    gx_val = (
        float(gx_override)
        if gx_override is not None
        else (
            float(getattr(fit, "gamma_exog", 0.0))
            if (fit is not None and getattr(fit, "gamma_exog", None) is not None)
            else None
        )
    )
    gi_val = float(getattr(fit, "gamma_intercept", 0.0)) if fit is not None else 0.0
    exog_lag_val = (
        int(getattr(fit, "exog_lag")) if (fit is not None and getattr(fit, "exog_lag", None) is not None) else None
    )
    bkps_list = (
        list(getattr(fit, "breakpoints", []))
        if fit is not None
        else list(st.session_state.get("detected_breakpoints", []) or [])
    )

    if (k_val is not None) and (r_list is not None) and bkps_list:
        payload["fit_params"] = {
            "carrying_capacity": int(round(float(k_val))),
            "segment_growth_rates": [round(float(x), 6) for x in r_list],
            "breakpoints": list(bkps_list),
            "gamma_pulse": _rf(gp_val),
            "gamma_step": _rf(gs_val),
            "gamma_exog": _rf(gx_val),
            "gamma_intercept": _rf(gi_val),
            "exog_lag": exog_lag_val,
        }

    return json.dumps(payload, indent=2).encode("utf-8")


def apply_phase_one_json(file_like) -> None:
    """Load a Phase 1 JSON artifact and update session_state accordingly.

    - Restores series (import_total/import_paid)
    - Restores events_df
    - Restores ad_spend (covariates_df)
    - Restores detection outputs and feature knobs
    - Rebuilds features_df (pulse, step, adstock, ad_effect_log) on the union monthly index
    """
    # Read bytes
    data: bytes
    if isinstance(file_like, (bytes, bytearray)):
        data = bytes(file_like)
    else:
        data = file_like.read()

    obj = json.loads(data)

    # Series
    total_s = _records_to_series(obj.get("total_series"))
    paid_s = _records_to_series(obj.get("paid_series"))
    if total_s is not None:
        st.session_state["import_total"] = total_s.sort_index()
    if paid_s is not None:
        st.session_state["import_paid"] = paid_s.sort_index()

    # Detection outputs / provenance
    st.session_state["detected_breakpoints"] = list(obj.get("breakpoints_indices", []) or [])
    st.session_state["detected_change_dates"] = [
        pd.to_datetime(d).to_period("M").to_timestamp("M") for d in (obj.get("breakpoints_dates", []) or [])
    ]
    # Expect canonical codes only
    st.session_state["detect_on"] = str(obj.get("detect_mode", "auto")).strip().lower()
    st.session_state["detected_target_label"] = obj.get("detected_target_label")
    st.session_state["detected_target_col"] = obj.get("target_col_for_fit")

    # Events
    ev_rows = obj.get("events") or []
    if isinstance(ev_rows, list) and ev_rows:
        ev_df = pd.DataFrame(ev_rows)
        if not ev_df.empty:
            with suppress(Exception):
                ev_df["date"] = pd.to_datetime(ev_df["date"]).dt.date
            st.session_state["events_df"] = ev_df

    # Ad spend (covariates)
    ad_spend_rec = obj.get("ad_spend") or []
    covariates_df = None
    if isinstance(ad_spend_rec, list) and ad_spend_rec:
        cov = pd.DataFrame(ad_spend_rec)
        if {"date", "spend"}.issubset(cov.columns):
            cov["date"] = pd.to_datetime(cov["date"]).dt.to_period("M").dt.to_timestamp("M")
            covariates_df = cov.set_index("date").rename(columns={"spend": "ad_spend"}).sort_index()
            st.session_state["covariates_df"] = covariates_df

    # Feature knobs
    st.session_state["adstock_lambda"] = float(obj.get("adstock_lambda", 0.5))
    st.session_state["ad_log_theta"] = float(obj.get("ad_log_theta", 500.0))

    # Rebuild features_df on union monthly index
    idx_sources: list[pd.DatetimeIndex] = []
    if isinstance(st.session_state.get("import_total"), pd.Series) and not st.session_state["import_total"].empty:
        idx_sources.append(st.session_state["import_total"].index)
    if isinstance(st.session_state.get("import_paid"), pd.Series) and not st.session_state["import_paid"].empty:
        idx_sources.append(st.session_state["import_paid"].index)
    if idx_sources:
        base_index = idx_sources[0]
        for idx in idx_sources[1:]:
            base_index = base_index.union(idx)
        base_index = base_index.sort_values()
    else:
        base_index = pd.period_range("2000-01", periods=1, freq="M").to_timestamp("M")

    # Build pulse/step from events
    pulse = pd.Series(0.0, index=base_index, name="pulse")
    step = pd.Series(0.0, index=base_index, name="step")
    ev_df2 = st.session_state.get("events_df")
    if isinstance(ev_df2, pd.DataFrame) and not ev_df2.empty:
        e2 = ev_df2.dropna(subset=["date"]).copy()
        with suppress(Exception):
            e2["date"] = pd.to_datetime(e2["date"]).dt.to_period("M").dt.to_timestamp("M")
        e2 = e2.sort_values("date")
        e2 = e2.drop_duplicates(subset=["date", "persistence"], keep="first")
        for _, r in e2.iterrows():
            d = r.get("date")
            if pd.isna(d) or d not in base_index:
                continue
            cost = float(r.get("cost", 1.0) or 1.0)
            kind = str(r.get("type", ""))
            weight = cost if kind.lower() in {"ad spend", "ad"} else 1.0
            persistence = str(r.get("persistence", "")).strip().lower()
            if persistence == "no effect":
                continue
            if persistence == "persistent":
                step.loc[d:] += 1.0
            elif persistence == "transient":
                pulse.loc[d] += float(weight)

    # Adstock + log response
    ad_spend = pd.Series(0.0, index=base_index, name="ad_spend")
    if covariates_df is not None:
        ad_spend = covariates_df.reindex(base_index).fillna(0.0)["ad_spend"]
    lam = float(st.session_state.get("adstock_lambda", 0.5))
    theta = float(st.session_state.get("ad_log_theta", 500.0))
    adstock_vals: list[float] = []
    prev = 0.0
    for v in ad_spend.to_list():
        s_val = float(v) + lam * float(prev)
        adstock_vals.append(s_val)
        prev = s_val
    adstock = pd.Series(adstock_vals, index=base_index, name="adstock")
    ad_effect_log = pd.Series(
        (adstock / max(theta, 1e-9)).add(1.0).apply(lambda x: float(math.log(x))),
        index=base_index,
        name="ad_effect_log",
    )

    st.session_state["features_df"] = pd.DataFrame(
        {
            "pulse": pulse,
            "step": step,
            "adstock": adstock,
            "ad_effect_log": ad_effect_log,
        }
    )
