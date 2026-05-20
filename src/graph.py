from __future__ import annotations

import os
import re
from html import escape
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents import (
    run_audit_agent,
    run_context_agent,
    run_ops_agent,
    run_planner_agent,
    run_report_agent,
)
from tools.csv_tools import analyze_csv
from tools.email_tools import send_email_smtp
from tools.knowledge_tools import load_reference_facts, run_rag_eval
from tools.pdf_tools import KnowledgeRag, PdfRag, docs_to_snippets
from tools.weather_tools import derive_dispatch_weather_risk, get_weather_forecast

load_dotenv()


class AppState(TypedDict, total=False):
    pdf_path: str
    csv_path: str
    knowledge_sources: List[str]
    reference_markdown_path: str
    max_audit_retries: int

    business_context: str
    retrieved_policy_context: str
    retrieved_reference_context: str
    structured_reference_facts: Dict[str, Any]
    rag_eval_results: Dict[str, Any]

    csv_summary: Dict[str, Any]
    csv_kpis: Dict[str, Any]
    dq_audit_log: List[Dict[str, Any]]
    audit_log_preview: str
    anomalies_md: str
    trend_analysis: Dict[str, Any]
    corridor_kpis: List[Dict[str, Any]]
    sla_risk_flags: List[str]
    ops_insights: str

    weather_risk: Dict[str, Any]

    planner_draft: Dict[str, Any]
    planner_attempts: int
    audit_result: Dict[str, Any]
    audit_feedback: Dict[str, Any]
    report_html: str
    appendix_text: str


def _existing_paths(paths: List[str]) -> List[str]:
    return [path for path in paths if Path(path).exists()]


def default_knowledge_sources(state: AppState) -> List[str]:
    requested_sources = state.get("knowledge_sources")
    if requested_sources:
        return _existing_paths(requested_sources)

    candidates = [
        state.get("pdf_path", "data/SeeWeeS Specialty Dispatch Playbook.pdf"),
        "data-for-enhancement/SeeWeeS Specialty Dispatch Playbook.md",
        "data/About SeeWeeS Specialty distribution.pdf",
    ]
    return _existing_paths(candidates)


def _find_reference_markdown_path(state: AppState) -> str | None:
    explicit = state.get("reference_markdown_path")
    if explicit and Path(explicit).exists():
        return explicit
    for source in state.get("knowledge_sources", []):
        if source.lower().endswith(".md") and Path(source).exists():
            return source
    candidate = Path("data-for-enhancement/SeeWeeS Specialty Dispatch Playbook.md")
    return str(candidate) if candidate.exists() else None


def _render_preview_table(rows: List[Dict[str, Any]], limit: int = 10) -> str:
    if not rows:
        return "(none)"
    import pandas as pd

    return pd.DataFrame(rows[:limit]).to_markdown(index=False)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


OUT_OF_SCOPE_CONTEXT_PATTERNS = (
    r"## 8\. Truck Capacity & Packing Model.*?(?=\n## |\Z)",
    r"## 13\. Resource Constraints and Allocation Policy.*?(?=\n## |\Z)",
    r"- Recommended resource allocation \(drivers/trucks/temp-controlled\).*?(?=\n- |\n## |\Z)",
)


FORBIDDEN_PLANNER_OUTPUT_TERMS = (
    "truck capacity",
    "packing constraint",
    "resource allocation",
    "driver allocation",
    "truck allocation",
    "driver/truck",
    "final dispatch report will include",
    "final report will include",
    "this report will include",
    "ensuring full compliance",
    "as mandated by policy",
)


PLANNER_OUTPUT_REPLACEMENTS = (
    (r"\btruck[- ]capacity compliance\b", "current-scope policy alignment"),
    (r"\btruck[- ]capacity\b", "current-scope logistics"),
    (r"\bstricter SLA tiering\b", "corridor-level SLA monitoring"),
    (r"\bstricter SLA tiers\b", "corridor-level SLA monitoring"),
    (r"\bspecial SLA rules\b", "corridor-level SLA rules"),
    (r"\bspecial escalation rules\b", "rule-based escalation triggers"),
    (r"\bmedicine[- ]type[- ]specific escalation\b", "rule-code monitoring"),
    (r"\bdriven by\b", "associated with"),
    (r"\bwill continue to\b", "currently"),
    (r"\bwill continue\b", "currently continues"),
    (r"\bpacking constraints\b", "item handling requirements"),
    (r"\bpacking constraint\b", "item handling requirement"),
    (r"\bpackaging attributes\b", "item handling attributes"),
    (r"\bproduct-class constraints\b", "product handling requirements"),
    (r"\bresource allocation\b", "current-scope dispatch monitoring"),
    (r"\bdriver allocation\b", "current-scope dispatch monitoring"),
    (r"\btruck allocation\b", "current-scope dispatch monitoring"),
    (r"\bdriver/truck sufficiency\b", "current-scope dispatch readiness"),
    (r"\bdriver/truck\b", "dispatch"),
    (r"\bcapacity use\b", "planning exposure"),
    (r"\bfinal dispatch report will include\b", "recommendation includes"),
    (r"\bfinal report will include\b", "recommendation includes"),
    (r"\bthis report will include\b", "recommendation includes"),
    (r"\bensuring full compliance\b", "supporting policy alignment"),
    (r"\bas mandated by policy\b", "per retrieved playbook rules"),
)


def _filter_implemented_scope_context(text: str) -> str:
    scoped = text
    for pattern in OUT_OF_SCOPE_CONTEXT_PATTERNS:
        scoped = re.sub(pattern, "", scoped, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", scoped).strip()


def _sanitize_planner_output(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_planner_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_planner_output(item) for item in value]
    if not isinstance(value, str):
        return value

    sanitized = value
    for pattern, replacement in PLANNER_OUTPUT_REPLACEMENTS:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized


def _html_table(rows: List[Dict[str, Any]], columns: List[str] | None = None) -> str:
    if not rows:
        return "<p>(none)</p>"

    if columns is None:
        columns = list(rows[0].keys())

    header = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{escape('' if row.get(column) is None else str(row.get(column)))}</td>"
            for column in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        "<table border='1' cellspacing='0' cellpadding='6' style='border-collapse:collapse;width:100%;margin:12px 0;'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _true_risk_flags(flags: Dict[str, Any] | None) -> str:
    if not flags:
        return "none"
    true_flags = [key for key, value in flags.items() if value is True and key != "weather_data_unavailable"]
    return ", ".join(true_flags) if true_flags else "none"


def _render_weather_snapshot_html(weather_risk: Dict[str, Any]) -> str:
    per_waypoint = weather_risk.get("per_waypoint", [])
    if not per_waypoint:
        return ""

    worst_waypoint = weather_risk.get("worst_waypoint", {}).get("waypoint")
    rows: list[dict[str, Any]] = []
    for waypoint in per_waypoint:
        waypoint_id = waypoint.get("waypoint", "")
        if waypoint_id == worst_waypoint:
            waypoint_id = f"{waypoint_id} (worst)"
        rows.append(
            {
                "Waypoint": waypoint_id,
                "City": waypoint.get("city", ""),
                "Risk Score": waypoint.get("risk_score_0_3", ""),
                "Precip mm": waypoint.get("max_precip_mm_day", ""),
                "Wind km/h": waypoint.get("max_wind_gust_kmh", ""),
                "Min Temp C": waypoint.get("min_temp_c", ""),
                "Risk Flags": _true_risk_flags(waypoint.get("risk_flags")),
            }
        )

    return (
        "<h2>4. Weather Route Snapshot</h2>"
        "<p>Per-waypoint weather evidence used for the corridor buffer decision. "
        "Scores map to the playbook buffer tiers: 0 = 0%, 1 = 10%, 2 = 25%, 3 = 40% plus escalation.</p>"
        + _html_table(rows, ["Waypoint", "City", "Risk Score", "Precip mm", "Wind km/h", "Min Temp C", "Risk Flags"])
    )


def _ensure_weather_snapshot_section(report_html: str, weather_risk: Dict[str, Any]) -> str:
    if "Weather Route Snapshot" in report_html:
        return report_html

    weather_html = _render_weather_snapshot_html(weather_risk)
    if not weather_html:
        return report_html

    corridor_heading = re.search(r"<h2>\s*\d+\.\s*Corridor Performance Snapshot\s*</h2>", report_html, re.IGNORECASE)
    if corridor_heading:
        return report_html[: corridor_heading.start()] + weather_html + report_html[corridor_heading.start() :]
    return f"{report_html}\n{weather_html}"


def _render_deep_dive_html(state: AppState) -> str:
    trend = state.get("trend_analysis", {})
    deep = trend.get("deep_dive_tables", {})
    if not deep:
        return ""

    sections: list[str] = ["<section id='deep-dive-appendix'><h2>Deep-Dive Analytics Appendix</h2>"]

    daily_valid = deep.get("daily_valid_units", [])
    if daily_valid:
        sections.append("<h3>Daily Valid Shipment Trend</h3>")
        sections.append(_html_table(daily_valid, ["shipment_date", "planning_day", "valid_units"]))

    corridor_day = deep.get("corridor_day_breakdown", [])
    if corridor_day:
        sections.append("<h3>Planning Window Corridor by Day Breakdown</h3>")
        sections.append(_html_table(corridor_day, ["corridor_id", "planning_day", "valid_units"]))

    item_spikes = deep.get("item_spikes", [])
    if item_spikes:
        sections.append("<h3>Top Item Spikes vs Historical Baseline</h3>")
        sections.append(
            _html_table(
                item_spikes,
                ["canonical_item_name", "planning_window_units", "historical_avg_daily_units", "spike_ratio"],
            )
        )

    correction_breakdown = deep.get("correction_breakdown", [])
    if correction_breakdown:
        sections.append("<h3>Resolution / Correction Breakdown</h3>")
        sections.append(_html_table(correction_breakdown, ["resolution_code", "units"]))

    exclusion_breakdown = deep.get("exclusion_breakdown", [])
    if exclusion_breakdown:
        sections.append("<h3>Exclusion Breakdown by Reason</h3>")
        sections.append(_html_table(exclusion_breakdown, ["reason_code", "units"]))

    daily_excluded = deep.get("daily_excluded_units", [])
    if daily_excluded:
        sections.append("<h3>Excluded Rows by Day and Reason</h3>")
        sections.append(_html_table(daily_excluded, ["shipment_date", "planning_day", "reason_code", "excluded_units"]))

    corrected_samples = deep.get("corrected_samples", [])
    if corrected_samples:
        sections.append("<h3>Sample Corrected Rows</h3>")
        sections.append(
            _html_table(
                corrected_samples,
                ["item_id", "item_name", "canonical_item_id", "canonical_item_name", "resolution_code", "planning_day", "corridor_id"],
            )
        )

    unresolved_samples = deep.get("unresolved_samples", [])
    if unresolved_samples:
        sections.append("<h3>Sample Excluded / Unresolved Rows</h3>")
        sections.append(
            _html_table(
                unresolved_samples,
                ["item_id", "item_name", "unique_item_id", "reason_code", "issue_codes", "planning_day", "corridor_id"],
            )
        )

    sections.append("</section>")
    return "".join(sections)


def _text_table(rows: List[Dict[str, Any]], columns: List[str] | None = None) -> str:
    if not rows:
        return "(none)"

    if columns is None:
        columns = list(rows[0].keys())

    safe_rows = [
        {column: "" if row.get(column) is None else str(row.get(column)) for column in columns}
        for row in rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in safe_rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(row[column].ljust(widths[column]) for column in columns)
        for row in safe_rows
    ]
    return "\n".join([header, divider, *body])


def _render_deep_dive_text(state: AppState) -> str:
    trend = state.get("trend_analysis", {})
    deep = trend.get("deep_dive_tables", {})
    if not deep:
        return ""

    sections: list[str] = [
        "MSBA Ops Deep-Dive Analytics Appendix",
        "=" * 37,
        "",
        "This appendix contains deterministic tables used by the agents. It is separated from the executive email body to keep the report decision-focused.",
    ]

    table_specs = [
        ("Daily Valid Shipment Trend", "daily_valid_units", ["shipment_date", "planning_day", "valid_units"]),
        ("Planning Window Corridor by Day Breakdown", "corridor_day_breakdown", ["corridor_id", "planning_day", "valid_units"]),
        ("Top Item Spikes vs Historical Baseline", "item_spikes", ["canonical_item_name", "planning_window_units", "historical_avg_daily_units", "spike_ratio"]),
        ("Resolution / Correction Breakdown", "correction_breakdown", ["resolution_code", "units"]),
        ("Exclusion Breakdown by Reason", "exclusion_breakdown", ["reason_code", "units"]),
        ("Excluded Rows by Day and Reason", "daily_excluded_units", ["shipment_date", "planning_day", "reason_code", "excluded_units"]),
        ("Sample Corrected Rows", "corrected_samples", ["item_id", "item_name", "canonical_item_id", "canonical_item_name", "resolution_code", "planning_day", "corridor_id"]),
        ("Sample Excluded / Unresolved Rows", "unresolved_samples", ["item_id", "item_name", "unique_item_id", "reason_code", "issue_codes", "planning_day", "corridor_id"]),
    ]

    for title, key, columns in table_specs:
        rows = deep.get(key, [])
        if rows:
            sections.extend(["", title, "-" * len(title), _text_table(rows, columns)])

    return "\n".join(sections)


def node_knowledge(state: AppState) -> AppState:
    knowledge_sources = default_knowledge_sources(state)
    rag = KnowledgeRag(persist_dir="chroma_db")
    vectordb = rag.build(knowledge_sources)

    policy_query = (
        "Extract dispatch rules, SLA requirements, weather buffer policy, escalation criteria, "
        "reporting expectations, and exception handling."
    )
    reference_query = (
        "Retrieve appendix and reference-table content for canonical item mapping, alias handling, "
        "legacy IDs, data quality rules, reporting requirements, and KPI thresholds."
    )

    policy_docs = rag.retrieve(vectordb, policy_query, k=6, stream="policy")
    reference_docs = rag.retrieve(vectordb, reference_query, k=8, stream="reference")

    policy_context = docs_to_snippets(policy_docs)
    reference_context = docs_to_snippets(reference_docs)
    business_context = run_context_agent(policy_context, reference_context)

    reference_markdown_path = _find_reference_markdown_path(
        {"knowledge_sources": knowledge_sources, "reference_markdown_path": state.get("reference_markdown_path", "")}
    )
    structured_reference_facts = (
        load_reference_facts(reference_markdown_path) if reference_markdown_path else {}
    )

    rag_eval_results = run_rag_eval(rag, vectordb, k=5)

    return {
        "knowledge_sources": knowledge_sources,
        "retrieved_policy_context": policy_context,
        "retrieved_reference_context": reference_context,
        "structured_reference_facts": structured_reference_facts,
        "business_context": business_context,
        "rag_eval_results": rag_eval_results,
    }


def node_trend_analysis(state: AppState) -> AppState:
    csv_path = state["csv_path"]
    result = analyze_csv(csv_path, reference_facts=state.get("structured_reference_facts", {}))

    anomalies_md = "(none detected)"
    if not result.anomalies.empty:
        anomalies_md = result.anomalies.head(10).to_markdown(index=False)

    dq_rows = result.audit_log.to_dict(orient="records")
    audit_log_preview = _render_preview_table(dq_rows, limit=12)
    ops_insights = run_ops_agent(
        summary=result.summary,
        kpis=result.kpis,
        trend_analysis=result.trend_analysis,
        audit_log_highlights=audit_log_preview,
        anomalies_md=anomalies_md,
    )

    return {
        "csv_summary": result.summary,
        "csv_kpis": result.kpis,
        "dq_audit_log": dq_rows,
        "audit_log_preview": audit_log_preview,
        "anomalies_md": anomalies_md,
        "trend_analysis": result.trend_analysis,
        "corridor_kpis": result.trend_analysis.get("corridor_kpis", []),
        "sla_risk_flags": result.trend_analysis.get("sla_risk_flags", []),
        "ops_insights": ops_insights,
    }


WAYPOINT_RE = re.compile(
    r"(W[1-9]\d*)\s+([A-Za-z][A-Za-z\s\-\.,]*)\s+([A-Z]{2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
    re.MULTILINE,
)


def _parse_waypoints_from_text(text: str) -> List[Dict[str, Any]]:
    waypoints: List[Dict[str, Any]] = []
    for match in WAYPOINT_RE.finditer(text):
        waypoint_id, city, state, lat, lon = match.groups()
        waypoints.append(
            {
                "id": waypoint_id.strip(),
                "city": city.strip(),
                "state": state.strip(),
                "lat": float(lat),
                "lon": float(lon),
            }
        )
    return waypoints


def _expected_buffer_pct(risk_score: int) -> int:
    return {0: 0, 1: 10, 2: 25, 3: 40}.get(risk_score, 0)


def _build_sla_risk_flags(state: AppState, weather_risk: Dict[str, Any]) -> List[str]:
    flags = list(state.get("sla_risk_flags", []))
    risk_score = int(
        weather_risk.get("risk_score_0_3", weather_risk.get("route_risk_score_0_3", 0)) or 0
    )
    buffer_pct = _expected_buffer_pct(risk_score)

    if risk_score > 0:
        for corridor in state.get("trend_analysis", {}).get("corridor_kpis", []):
            if corridor.get("default_sla_tier") == "Tier 1" and corridor.get("planning_window_valid_units", 0):
                flags.append(
                    f"Weather risk score {risk_score} requires a {buffer_pct}% travel buffer for Tier 1 corridor {corridor.get('corridor_id')}."
                )

    if risk_score == 3:
        flags.append("Weather risk score 3 requires escalation before final dispatch approval.")

    return list(dict.fromkeys(flags))


def _weather_unavailable_risk(error: Exception | str) -> Dict[str, Any]:
    return {
        "max_precip_mm_day": 0.0,
        "max_wind_gust_kmh": 0.0,
        "min_temp_c": None,
        "risk_flags": {
            "heavy_rain_risk": False,
            "high_wind_risk": False,
            "freezing_risk": False,
            "weather_data_unavailable": True,
        },
        "risk_score_0_3": 0,
        "weather_data_status": "unavailable",
        "weather_error": str(error),
    }


def _get_weather_risk_for_location(lat: str, lon: str, tz: str) -> Dict[str, Any]:
    try:
        forecast = get_weather_forecast(lat, lon, tz)
        risk = derive_dispatch_weather_risk(forecast)
        risk["weather_data_status"] = "available"
        return risk
    except Exception as exc:
        return _weather_unavailable_risk(exc)


def node_weather(state: AppState) -> AppState:
    tz = os.getenv("WEATHER_TZ", "America/New_York")
    pdf_path = state.get("pdf_path", "data/SeeWeeS Specialty Dispatch Playbook.pdf")
    if not Path(pdf_path).exists():
        pdf_path = ""

    if pdf_path:
        rag = PdfRag(persist_dir="chroma_db")
        vectordb = rag.build(pdf_path)
        docs = rag.retrieve(
            vectordb,
            (
                "Find the waypoint list for the I-95 corridor including W1, W2, W3, W4, W5 "
                "and their latitude and longitude."
            ),
            k=8,
        )
        waypoint_text = docs_to_snippets(docs)
        waypoints = _parse_waypoints_from_text(waypoint_text)
        if waypoints:
            worst: Dict[str, Any] | None = None
            per_waypoint: List[Dict[str, Any]] = []
            max_score = -1
            for waypoint in waypoints:
                risk = _get_weather_risk_for_location(str(waypoint["lat"]), str(waypoint["lon"]), tz)
                enriched = {"waypoint": waypoint["id"], "city": waypoint["city"], **risk}
                per_waypoint.append(enriched)
                if enriched.get("risk_score_0_3", -1) > max_score:
                    worst = enriched
                    max_score = enriched["risk_score_0_3"]

            if worst:
                weather_risk = {
                    "route_risk_score_0_3": max_score,
                    "worst_waypoint": worst,
                    "per_waypoint": per_waypoint,
                    "max_precip_mm_day": worst.get("max_precip_mm_day"),
                    "max_wind_gust_kmh": worst.get("max_wind_gust_kmh"),
                    "min_temp_c": worst.get("min_temp_c"),
                    "risk_flags": worst.get("risk_flags"),
                    "risk_score_0_3": worst.get("risk_score_0_3"),
                }
                return {
                    "weather_risk": weather_risk,
                    "sla_risk_flags": _build_sla_risk_flags(state, weather_risk),
                }

    lat = os.getenv("WEATHER_LAT", "40.7282")
    lon = os.getenv("WEATHER_LON", "-74.0776")
    weather_risk = _get_weather_risk_for_location(lat, lon, tz)
    return {
        "weather_risk": weather_risk,
        "sla_risk_flags": _build_sla_risk_flags(state, weather_risk),
    }


def node_planner(state: AppState) -> AppState:
    attempts = int(state.get("planner_attempts", 0)) + 1
    planner_draft = _sanitize_planner_output(run_planner_agent(
        business_context=state.get("business_context", ""),
        policy_context=_filter_implemented_scope_context(state.get("retrieved_policy_context", "")),
        reference_context=_filter_implemented_scope_context(state.get("retrieved_reference_context", "")),
        trend_analysis=state.get("trend_analysis", {}),
        corridor_kpis=state.get("corridor_kpis", []),
        sla_risk_flags=state.get("sla_risk_flags", []),
        kpis=state.get("csv_kpis", {}),
        ops_insights=state.get("ops_insights", ""),
        weather_risk=state.get("weather_risk", {}),
        audit_feedback=state.get("audit_feedback", {}),
    ))
    return {
        "planner_draft": planner_draft,
        "planner_attempts": attempts,
    }


def apply_deterministic_audit_checks(state: AppState, audit_result: Dict[str, Any]) -> Dict[str, Any]:
    weather_risk = state.get("weather_risk", {})
    risk_score = weather_risk.get("risk_score_0_3", weather_risk.get("route_risk_score_0_3", 0))
    expected_buffer = _expected_buffer_pct(int(risk_score or 0))
    planner_draft = state.get("planner_draft", {})

    violations = list(audit_result.get("violations", []))
    missing_evidence = list(audit_result.get("missing_evidence", []))
    revision_instructions = list(audit_result.get("revision_instructions", []))

    if planner_draft.get("recommended_buffer_pct") != expected_buffer:
        violations.append(
            f"Recommended buffer {planner_draft.get('recommended_buffer_pct')} does not match required buffer {expected_buffer} for risk score {risk_score}."
        )
        revision_instructions.append(
            f"Set recommended_buffer_pct to {expected_buffer} based on the risk-score policy."
        )

    if int(risk_score or 0) == 3 and not planner_draft.get("escalation_required"):
        violations.append("Risk score 3 requires escalation, but escalation_required is false.")
        revision_instructions.append("Set escalation_required to true when risk_score_0_3 is 3.")

    if not planner_draft.get("cited_rules"):
        missing_evidence.append("Planner draft does not cite supporting rules from retrieved policy context.")
        revision_instructions.append("Add cited_rules that reference the governing playbook rules used in the plan.")

    if state.get("sla_risk_flags") and not planner_draft.get("sla_risk_flags"):
        missing_evidence.append("Planner draft does not include deterministic SLA risk flags.")
        revision_instructions.append("Include the provided sla_risk_flags in the planner draft.")

    planner_text = str(planner_draft).lower()
    forbidden_hits = [term for term in FORBIDDEN_PLANNER_OUTPUT_TERMS if term in planner_text]
    if forbidden_hits:
        violations.append(
            "Planner draft includes terms reserved for internal reasoning or meta-reporting language: "
            + ", ".join(sorted(set(forbidden_hits)))
        )
        revision_instructions.append(
            "Remove truck-capacity/resource-allocation language and report-meta phrasing from all planner output fields."
        )

    audit_result["violations"] = violations
    audit_result["missing_evidence"] = missing_evidence
    audit_result["revision_instructions"] = revision_instructions
    audit_result["passed"] = bool(audit_result.get("passed", True)) and not violations and not missing_evidence
    return audit_result


def node_audit(state: AppState) -> AppState:
    base_audit_result = run_audit_agent(
        planner_draft=state.get("planner_draft", {}),
        business_context=state.get("business_context", ""),
        policy_context=_filter_implemented_scope_context(state.get("retrieved_policy_context", "")),
        reference_context=_filter_implemented_scope_context(state.get("retrieved_reference_context", "")),
        trend_analysis=state.get("trend_analysis", {}),
        corridor_kpis=state.get("corridor_kpis", []),
        sla_risk_flags=state.get("sla_risk_flags", []),
        kpis=state.get("csv_kpis", {}),
        weather_risk=state.get("weather_risk", {}),
        audit_log_summary=state.get("trend_analysis", {}).get("dq_summary", {}),
    )
    audit_result = apply_deterministic_audit_checks(state, base_audit_result)
    audit_feedback = {
        "passed": audit_result.get("passed", False),
        "violations": audit_result.get("violations", []),
        "unsupported_claims": audit_result.get("unsupported_claims", []),
        "missing_evidence": audit_result.get("missing_evidence", []),
        "revision_instructions": audit_result.get("revision_instructions", []),
    }
    return {
        "audit_result": audit_result,
        "audit_feedback": audit_feedback,
    }


def route_after_audit(state: AppState) -> str:
    audit_result = state.get("audit_result", {})
    if audit_result.get("passed"):
        return "report"
    max_retries = int(state.get("max_audit_retries", os.getenv("AUDIT_MAX_RETRIES", "2")))
    attempts = int(state.get("planner_attempts", 0))
    if attempts >= max_retries:
        return "report"
    return "planner"


def node_report(state: AppState) -> AppState:
    audit_result = state.get("audit_result", {})
    report_html = run_report_agent(
        business_context=state.get("business_context", ""),
        ops_insights=state.get("ops_insights", ""),
        trend_analysis=state.get("trend_analysis", {}),
        corridor_kpis=state.get("corridor_kpis", []),
        sla_risk_flags=state.get("sla_risk_flags", []),
        kpis=state.get("csv_kpis", {}),
        weather_risk=state.get("weather_risk", {}),
        planner_draft=state.get("planner_draft", {}),
        audit_result=audit_result,
        audit_log_preview=state.get("audit_log_preview", "(none)"),
    )
    report_html = _strip_code_fences(report_html)
    report_html = _ensure_weather_snapshot_section(report_html, state.get("weather_risk", {}))
    appendix_text = _render_deep_dive_text(state)

    if not audit_result.get("passed"):
        unresolved = audit_result.get("violations", []) + audit_result.get("missing_evidence", [])
        warning_html = (
            "<h2>Review Status</h2>"
            "<p><strong>Human review required before dispatch approval.</strong> "
            "The automated checks found items that need confirmation.</p>"
            f"<ul>{''.join(f'<li>{item}</li>' for item in unresolved[:8])}</ul>"
        )
        report_html = f"{report_html}\n{warning_html}"

    return {"report_html": report_html, "appendix_text": appendix_text}


def node_email(state: AppState) -> AppState:
    to_email = os.getenv("REPORT_EMAIL_TO", "").strip()
    if not to_email:
        return {}

    subject = "MSBA Ops Multi-Agent Dispatch Report"
    attachments = []
    if state.get("appendix_text"):
        attachments.append(
            {
                "filename": "msba_ops_deep_dive_appendix.txt",
                "mime_type": "text/plain",
                "content": state["appendix_text"],
            }
        )
    send_email_smtp(
        subject=subject,
        html_body=state["report_html"],
        to_email=to_email,
        attachments=attachments,
    )
    return {}


def build_graph():
    graph = StateGraph(AppState)

    graph.add_node("knowledge", node_knowledge)
    graph.add_node("trend_analysis", node_trend_analysis)
    graph.add_node("weather", node_weather)
    graph.add_node("planner", node_planner)
    graph.add_node("audit", node_audit)
    graph.add_node("report", node_report)
    graph.add_node("email", node_email)

    graph.set_entry_point("knowledge")
    graph.add_edge("knowledge", "trend_analysis")
    graph.add_edge("trend_analysis", "weather")
    graph.add_edge("weather", "planner")
    graph.add_edge("planner", "audit")
    graph.add_conditional_edges(
        "audit",
        route_after_audit,
        {
            "planner": "planner",
            "report": "report",
        },
    )
    graph.add_edge("report", "email")
    graph.add_edge("email", END)

    return graph.compile()
