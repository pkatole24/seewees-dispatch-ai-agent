from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _normalize_item_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalize_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


@dataclass
class CsvAnalysisResult:
    summary: Dict[str, Any]
    kpis: Dict[str, Any]
    anomalies: pd.DataFrame
    cleaned_shape: Tuple[int, int]
    numeric_cols: List[str]
    valid_shipments: pd.DataFrame
    excluded_shipments: pd.DataFrame
    audit_log: pd.DataFrame
    trend_analysis: Dict[str, Any]


def _resolve_reference_match(
    row: pd.Series,
    reference_facts: Dict[str, Any],
) -> tuple[dict[str, str] | None, str, list[str]]:
    item_id = _normalize_item_id(row.get("item_id"))
    item_name = _normalize_name(row.get("item_name"))
    exact_name_map = reference_facts.get("exact_name_map", {})
    canonical_by_id = reference_facts.get("canonical_by_id", {})
    canonical_by_canonical_id = reference_facts.get("canonical_by_canonical_id", {})
    alias_by_name = reference_facts.get("alias_by_name", {})
    legacy_by_item_id = reference_facts.get("legacy_by_item_id", {})

    notes: list[str] = []
    exact_match = exact_name_map.get(item_id, {}).get(item_name)
    if exact_match:
        return exact_match, "exact_match", notes

    alias_match = alias_by_name.get(item_name)
    if alias_match:
        canonical_item = canonical_by_canonical_id.get(alias_match["canonical_item_id"])
        if canonical_item:
            notes.append("resolved_from_alias_table")
            return canonical_item, "alias_match", notes

    legacy_match = legacy_by_item_id.get(item_id)
    if legacy_match:
        canonical_item = canonical_by_canonical_id.get(legacy_match["canonical_item_id"])
        if canonical_item:
            notes.append("resolved_from_legacy_id_map")
            return canonical_item, "legacy_id_map", notes

    candidates = canonical_by_id.get(item_id, [])
    if len(candidates) == 1:
        notes.append("single_candidate_item_id")
        return candidates[0], "exact_match", notes

    if len(candidates) > 1:
        notes.append("ambiguous_item_id_requires_name_match")
        return None, "excluded_unresolved", notes

    notes.append("item_id_not_found_in_reference")
    return None, "excluded_unresolved", notes


def _prepare_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all").copy()

    if "shipment_date" in df.columns:
        df["shipment_date"] = pd.to_datetime(df["shipment_date"], errors="coerce")
    if "is_planning_window" in df.columns:
        df["is_planning_window"] = pd.to_numeric(df["is_planning_window"], errors="coerce").fillna(0).astype(int)
    return df


def _build_reconciled_dataframe(df: pd.DataFrame, reference_facts: Dict[str, Any]) -> pd.DataFrame:
    duplicate_mask = pd.Series(False, index=df.index)
    if "unique_item_id" in df.columns:
        duplicate_mask = df["unique_item_id"].notna() & df["unique_item_id"].duplicated(keep=False)

    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        canonical_item, resolution_code, notes = _resolve_reference_match(row, reference_facts)
        issue_codes: list[str] = []
        inclusion_status = "valid"
        reason_code = resolution_code

        unique_item_id = row.get("unique_item_id")
        if pd.isna(unique_item_id) or str(unique_item_id).strip() == "":
            inclusion_status = "excluded"
            reason_code = "excluded_missing_unique_item_id"
            issue_codes.append("DQ-01")

        if bool(duplicate_mask.loc[idx]):
            inclusion_status = "excluded"
            reason_code = "excluded_duplicate_unique_item_id"
            issue_codes.append("DQ-04")

        if canonical_item is None:
            inclusion_status = "excluded"
            reason_code = "excluded_unresolved"
            if "DQ-02" not in issue_codes:
                issue_codes.append("DQ-02")

        raw_item_name = row.get("item_name")
        canonical_name = canonical_item.get("canonical_item_name") if canonical_item else None
        if canonical_item and _normalize_name(raw_item_name) != _normalize_name(canonical_name):
            issue_codes.append("DQ-03")

        record = row.to_dict()
        record.update(
            {
                "normalized_item_id": _normalize_item_id(row.get("item_id")),
                "normalized_item_name": _normalize_name(raw_item_name),
                "canonical_item_id": canonical_item.get("canonical_item_id") if canonical_item else None,
                "canonical_item_name": canonical_name,
                "medicine_type": canonical_item.get("medicine_type") if canonical_item else None,
                "temp_control": canonical_item.get("temp_control") if canonical_item else None,
                "product_class": canonical_item.get("product_class") if canonical_item else None,
                "resolution_code": resolution_code,
                "reason_code": reason_code,
                "inclusion_status": inclusion_status,
                "issue_codes": issue_codes,
                "resolution_notes": notes,
                "was_corrected": resolution_code in {"alias_match", "legacy_id_map"},
            }
        )
        records.append(record)

    reconciled = pd.DataFrame(records)
    if "shipment_date" in reconciled.columns:
        reconciled["shipment_date"] = pd.to_datetime(reconciled["shipment_date"], errors="coerce")
    return reconciled


def _series_to_json_ready(series: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in series.items():
        if pd.isna(value):
            result[str(key)] = None
        elif isinstance(value, (np.integer, int)):
            result[str(key)] = int(value)
        elif isinstance(value, (np.floating, float)):
            result[str(key)] = float(value)
        else:
            result[str(key)] = value
    return result


def _frame_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    safe = df.copy()
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = safe[column].dt.strftime("%Y-%m-%d")
    return safe.where(pd.notna(safe), None).to_dict(orient="records")


def _planning_mask(df: pd.DataFrame) -> pd.Series:
    if "is_planning_window" in df.columns:
        return df["is_planning_window"].fillna(0).astype(int) == 1
    if "planning_day" in df.columns:
        return df["planning_day"].astype(str).isin(["Day0", "Day1"])
    return pd.Series(False, index=df.index)


def _build_anomalies(valid_shipments: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"shipment_date", "canonical_item_name"}
    if not required_columns.issubset(valid_shipments.columns):
        return pd.DataFrame()

    planning = valid_shipments[_planning_mask(valid_shipments)].copy()
    history = valid_shipments[~_planning_mask(valid_shipments)].copy()
    if planning.empty or history.empty:
        return pd.DataFrame()

    history_daily = (
        history.groupby(["canonical_item_name", "shipment_date"]).size().groupby("canonical_item_name").mean()
    )
    planning_counts = planning.groupby("canonical_item_name").size()

    anomaly_rows: list[dict[str, Any]] = []
    for item_name, planning_count in planning_counts.items():
        baseline = float(history_daily.get(item_name, 0.0))
        ratio = None if baseline == 0 else round(float(planning_count) / baseline, 2)
        if baseline == 0 or (ratio is not None and ratio >= 1.5):
            anomaly_rows.append(
                {
                    "canonical_item_name": item_name,
                    "planning_window_units": int(planning_count),
                    "historical_avg_daily_units": round(baseline, 2),
                    "spike_ratio": ratio,
                }
            )

    return pd.DataFrame(anomaly_rows).sort_values(
        by=["planning_window_units", "historical_avg_daily_units"],
        ascending=[False, False],
    )


def _corridor_tier_map(reference_facts: Dict[str, Any] | None) -> dict[str, str]:
    corridor_by_id = (reference_facts or {}).get("corridor_by_id", {})
    return {
        str(corridor_id): str(row.get("default_sla_tier", "Unknown") or "Unknown")
        for corridor_id, row in corridor_by_id.items()
    }


def _build_corridor_kpis(
    reconciled: pd.DataFrame,
    reference_facts: Dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    if "corridor_id" not in reconciled.columns:
        return [], {}, []

    tier_by_corridor = _corridor_tier_map(reference_facts)
    planning_rows = reconciled[_planning_mask(reconciled)].copy()
    if planning_rows.empty:
        return [], {}, []

    corridor_ids = sorted(str(value) for value in planning_rows["corridor_id"].dropna().unique())
    corridor_kpis: list[dict[str, Any]] = []
    tier_mix: dict[str, int] = {}
    sla_risk_flags: list[str] = []

    for corridor_id in corridor_ids:
        corridor_rows = planning_rows[planning_rows["corridor_id"].astype(str) == corridor_id]
        valid_rows = corridor_rows[corridor_rows["inclusion_status"] == "valid"]
        excluded_rows = corridor_rows[corridor_rows["inclusion_status"] != "valid"]
        valid_units = int(len(valid_rows))
        excluded_units = int(len(excluded_rows))
        total_units = valid_units + excluded_units
        excluded_rate = round(excluded_units / max(total_units, 1), 3)
        sla_tier = tier_by_corridor.get(corridor_id, "Unknown")
        tier_mix[sla_tier] = tier_mix.get(sla_tier, 0) + valid_units

        corridor_kpis.append(
            {
                "corridor_id": corridor_id,
                "default_sla_tier": sla_tier,
                "planning_window_valid_units": valid_units,
                "planning_window_excluded_units": excluded_units,
                "planning_window_total_units": total_units,
                "planning_window_excluded_rate": excluded_rate,
            }
        )

        if sla_tier == "Tier 1" and valid_units:
            sla_risk_flags.append(
                f"{corridor_id} is a Tier 1 corridor with {valid_units} valid planning-window unit(s); monitor 6-hour SLA exposure."
            )
        if sla_tier == "Tier 1" and excluded_units:
            sla_risk_flags.append(
                f"{corridor_id} has {excluded_units} excluded planning-window unit(s), creating traceability risk for Tier 1 dispatch planning."
            )

    return corridor_kpis, tier_mix, sla_risk_flags


def _build_trend_analysis(
    reconciled: pd.DataFrame,
    reference_facts: Dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    valid_shipments = reconciled[reconciled["inclusion_status"] == "valid"].copy()
    excluded_shipments = reconciled[reconciled["inclusion_status"] != "valid"].copy()

    planning_mask = _planning_mask(valid_shipments)
    planning = valid_shipments[planning_mask].copy()
    history = valid_shipments[~planning_mask].copy()

    history_daily_counts = (
        history.groupby("shipment_date").size() if "shipment_date" in history.columns and not history.empty else pd.Series(dtype=float)
    )
    planning_daily_counts = (
        planning.groupby("shipment_date").size() if "shipment_date" in planning.columns and not planning.empty else pd.Series(dtype=float)
    )

    history_avg_daily = float(history_daily_counts.mean()) if not history_daily_counts.empty else 0.0
    planning_total = int(len(planning))
    planning_horizon_days = int(planning["shipment_date"].nunique()) if "shipment_date" in planning.columns else 0
    planning_avg_daily = round(planning_total / planning_horizon_days, 2) if planning_horizon_days else 0.0

    corridor_mix = {}
    if "corridor_id" in planning.columns and not planning.empty:
        corridor_mix = planning.groupby("corridor_id").size().astype(int).to_dict()

    item_mix_planning = (
        planning.groupby("canonical_item_name").size().sort_values(ascending=False).head(10)
        if not planning.empty
        else pd.Series(dtype=int)
    )
    item_mix_history = (
        history.groupby("canonical_item_name").size().sort_values(ascending=False).head(10)
        if not history.empty
        else pd.Series(dtype=int)
    )
    correction_breakdown = (
        valid_shipments.groupby(["resolution_code"]).size().reset_index(name="units")
        if not valid_shipments.empty
        else pd.DataFrame(columns=["resolution_code", "units"])
    )
    exclusion_breakdown = (
        excluded_shipments.groupby(["reason_code"]).size().reset_index(name="units")
        if not excluded_shipments.empty
        else pd.DataFrame(columns=["reason_code", "units"])
    )
    daily_valid_units = (
        valid_shipments.groupby(["shipment_date", "planning_day"]).size().reset_index(name="valid_units")
        if {"shipment_date", "planning_day"}.issubset(valid_shipments.columns)
        else pd.DataFrame(columns=["shipment_date", "planning_day", "valid_units"])
    )
    daily_excluded_units = (
        excluded_shipments.groupby(["shipment_date", "planning_day", "reason_code"]).size().reset_index(name="excluded_units")
        if {"shipment_date", "planning_day", "reason_code"}.issubset(excluded_shipments.columns)
        else pd.DataFrame(columns=["shipment_date", "planning_day", "reason_code", "excluded_units"])
    )
    corridor_day_breakdown = (
        planning.groupby(["corridor_id", "planning_day"]).size().reset_index(name="valid_units")
        if {"corridor_id", "planning_day"}.issubset(planning.columns)
        else pd.DataFrame(columns=["corridor_id", "planning_day", "valid_units"])
    )
    corridor_item_breakdown = (
        planning.groupby(["corridor_id", "canonical_item_name"]).size().reset_index(name="units")
        .sort_values(["corridor_id", "units", "canonical_item_name"], ascending=[True, False, True])
        if {"corridor_id", "canonical_item_name"}.issubset(planning.columns)
        else pd.DataFrame(columns=["corridor_id", "canonical_item_name", "units"])
    )
    item_spike_table = pd.DataFrame(columns=["canonical_item_name", "planning_window_units", "historical_avg_daily_units", "spike_ratio"])
    if not planning.empty:
        planning_item_counts = planning.groupby("canonical_item_name").size().rename("planning_window_units")
        history_item_daily = (
            history.groupby(["canonical_item_name", "shipment_date"]).size().groupby("canonical_item_name").mean()
            if not history.empty
            else pd.Series(dtype=float)
        )
        item_spike_table = planning_item_counts.reset_index()
        item_spike_table["historical_avg_daily_units"] = item_spike_table["canonical_item_name"].map(history_item_daily).fillna(0.0)
        item_spike_table["spike_ratio"] = item_spike_table.apply(
            lambda row: None
            if float(row["historical_avg_daily_units"]) == 0.0
            else round(float(row["planning_window_units"]) / float(row["historical_avg_daily_units"]), 2),
            axis=1,
        )
        item_spike_table = item_spike_table.sort_values(
            ["planning_window_units", "historical_avg_daily_units"],
            ascending=[False, False],
        ).head(12)

    corrected_samples = valid_shipments[valid_shipments["was_corrected"]].copy()
    corrected_samples = corrected_samples[
        [
            column
            for column in [
                "item_id",
                "item_name",
                "canonical_item_id",
                "canonical_item_name",
                "resolution_code",
                "planning_day",
                "corridor_id",
            ]
            if column in corrected_samples.columns
        ]
    ].head(8)
    unresolved_samples = excluded_shipments.copy()
    unresolved_samples = unresolved_samples[
        [
            column
            for column in [
                "item_id",
                "item_name",
                "unique_item_id",
                "reason_code",
                "issue_codes",
                "planning_day",
                "corridor_id",
            ]
            if column in unresolved_samples.columns
        ]
    ].head(8)

    dq_summary = {
        "excluded_rows": int(len(excluded_shipments)),
        "excluded_rate": round(float(len(excluded_shipments)) / max(len(reconciled), 1), 3),
        "reason_counts": {
            str(key): int(value)
            for key, value in excluded_shipments["reason_code"].value_counts().to_dict().items()
        },
        "correction_counts": {
            str(key): int(value)
            for key, value in valid_shipments["resolution_code"].value_counts().to_dict().items()
        },
    }
    corridor_kpis, sla_tier_mix, sla_risk_flags = _build_corridor_kpis(reconciled, reference_facts)

    trend_analysis = {
        "period_over_period": {
            "history_avg_daily_valid_units": round(history_avg_daily, 2),
            "planning_window_total_valid_units": planning_total,
            "planning_window_avg_daily_valid_units": planning_avg_daily,
            "delta_vs_history_avg_daily": round(planning_avg_daily - history_avg_daily, 2),
        },
        "planning_window": {
            "days": planning_horizon_days,
            "valid_units": planning_total,
            "excluded_units": int(len(excluded_shipments[_planning_mask(excluded_shipments)])),
            "corridor_mix": corridor_mix,
            "sla_tier_mix": sla_tier_mix,
        },
        "corridor_kpis": corridor_kpis,
        "sla_risk_flags": sla_risk_flags,
        "dq_summary": dq_summary,
        "item_mix": {
            "planning_window_top_items": _series_to_json_ready(item_mix_planning),
            "history_top_items": _series_to_json_ready(item_mix_history),
        },
        "deep_dive_tables": {
            "daily_valid_units": _frame_to_records(daily_valid_units),
            "daily_excluded_units": _frame_to_records(daily_excluded_units),
            "corridor_day_breakdown": _frame_to_records(corridor_day_breakdown),
            "corridor_item_breakdown": _frame_to_records(corridor_item_breakdown),
            "correction_breakdown": _frame_to_records(correction_breakdown),
            "exclusion_breakdown": _frame_to_records(exclusion_breakdown),
            "item_spikes": _frame_to_records(item_spike_table),
            "corrected_samples": _frame_to_records(corrected_samples),
            "unresolved_samples": _frame_to_records(unresolved_samples),
        },
    }

    kpis = {
        "rows_count": int(len(reconciled)),
        "valid_units": int(len(valid_shipments)),
        "excluded_units": int(len(excluded_shipments)),
        "excluded_rate": dq_summary["excluded_rate"],
        "corrected_units": int(valid_shipments["was_corrected"].sum()),
        "planning_window_valid_units": planning_total,
        "history_valid_units": int(len(history)),
        "planning_window_avg_daily_valid_units": planning_avg_daily,
        "history_avg_daily_valid_units": round(history_avg_daily, 2),
    }

    if "corridor_id" in planning.columns and corridor_mix:
        kpis["planning_window_corridor_count"] = int(planning["corridor_id"].nunique())

    return trend_analysis, kpis


def analyze_csv(csv_path: str, reference_facts: Dict[str, Any] | None = None) -> CsvAnalysisResult:
    df = _prepare_dataframe(csv_path)
    original_shape = df.shape

    reference_facts = reference_facts or {}
    if reference_facts:
        reconciled = _build_reconciled_dataframe(df, reference_facts)
    else:
        reconciled = df.copy()
        reconciled["inclusion_status"] = "valid"
        reconciled["reason_code"] = "unreconciled"
        reconciled["resolution_code"] = "unreconciled"
        reconciled["was_corrected"] = False
        reconciled["issue_codes"] = [[] for _ in range(len(reconciled))]
        reconciled["resolution_notes"] = [[] for _ in range(len(reconciled))]

    summary = {
        "rows_original": int(original_shape[0]),
        "cols_original": int(original_shape[1]),
        "rows_after_drop_empty": int(df.shape[0]),
        "missingness_top": df.isna().mean().sort_values(ascending=False).head(10).to_dict(),
        "column_dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "columns": list(df.columns),
    }

    valid_shipments = reconciled[reconciled["inclusion_status"] == "valid"].copy()
    excluded_shipments = reconciled[reconciled["inclusion_status"] != "valid"].copy()
    trend_analysis, kpis = _build_trend_analysis(reconciled, reference_facts)
    anomalies = _build_anomalies(valid_shipments)
    numeric_cols = valid_shipments.select_dtypes(include=[np.number]).columns.tolist()

    audit_columns = [
        column
        for column in [
            "shipment_date",
            "planning_day",
            "corridor_id",
            "item_id",
            "item_name",
            "unique_item_id",
            "canonical_item_id",
            "canonical_item_name",
            "resolution_code",
            "reason_code",
            "inclusion_status",
            "issue_codes",
            "resolution_notes",
        ]
        if column in reconciled.columns
    ]
    audit_log = reconciled[audit_columns].copy()

    return CsvAnalysisResult(
        summary=summary,
        kpis=kpis,
        anomalies=anomalies,
        cleaned_shape=reconciled.shape,
        numeric_cols=numeric_cols,
        valid_shipments=valid_shipments,
        excluded_shipments=excluded_shipments,
        audit_log=audit_log,
        trend_analysis=trend_analysis,
    )
