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
    ops_insights: str

    weather_risk: Dict[str, Any]

    planner_draft: Dict[str, Any]
    planner_attempts: int
    audit_result: Dict[str, Any]
    audit_feedback: Dict[str, Any]
    report_html: str


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
                forecast = get_weather_forecast(str(waypoint["lat"]), str(waypoint["lon"]), tz)
                risk = derive_dispatch_weather_risk(forecast)
                enriched = {"waypoint": waypoint["id"], "city": waypoint["city"], **risk}
                per_waypoint.append(enriched)
                if enriched.get("risk_score_0_3", -1) > max_score:
                    worst = enriched
                    max_score = enriched["risk_score_0_3"]

            if worst:
                return {
                    "weather_risk": {
                        "route_risk_score_0_3": max_score,
                        "worst_waypoint": worst,
                        "per_waypoint": per_waypoint,
                        "max_precip_mm_day": worst.get("max_precip_mm_day"),
                        "max_wind_gust_kmh": worst.get("max_wind_gust_kmh"),
                        "min_temp_c": worst.get("min_temp_c"),
                        "risk_flags": worst.get("risk_flags"),
                        "risk_score_0_3": worst.get("risk_score_0_3"),
                    }
                }

    lat = os.getenv("WEATHER_LAT", "40.7282")
    lon = os.getenv("WEATHER_LON", "-74.0776")
    forecast = get_weather_forecast(lat, lon, tz)
    return {"weather_risk": derive_dispatch_weather_risk(forecast)}


def node_planner(state: AppState) -> AppState:
    attempts = int(state.get("planner_attempts", 0)) + 1
    planner_draft = run_planner_agent(
        business_context=state.get("business_context", ""),
        policy_context=state.get("retrieved_policy_context", ""),
        reference_context=state.get("retrieved_reference_context", ""),
        trend_analysis=state.get("trend_analysis", {}),
        kpis=state.get("csv_kpis", {}),
        ops_insights=state.get("ops_insights", ""),
        weather_risk=state.get("weather_risk", {}),
        audit_feedback=state.get("audit_feedback", {}),
    )
    return {
        "planner_draft": planner_draft,
        "planner_attempts": attempts,
    }


def apply_deterministic_audit_checks(state: AppState, audit_result: Dict[str, Any]) -> Dict[str, Any]:
    weather_risk = state.get("weather_risk", {})
    risk_score = weather_risk.get("risk_score_0_3", weather_risk.get("route_risk_score_0_3", 0))
    expected_buffer = {0: 0, 1: 10, 2: 25, 3: 40}.get(int(risk_score or 0), 0)
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

    audit_result["violations"] = violations
    audit_result["missing_evidence"] = missing_evidence
    audit_result["revision_instructions"] = revision_instructions
    audit_result["passed"] = bool(audit_result.get("passed", True)) and not violations and not missing_evidence
    return audit_result


def node_audit(state: AppState) -> AppState:
    base_audit_result = run_audit_agent(
        planner_draft=state.get("planner_draft", {}),
        business_context=state.get("business_context", ""),
        policy_context=state.get("retrieved_policy_context", ""),
        reference_context=state.get("retrieved_reference_context", ""),
        trend_analysis=state.get("trend_analysis", {}),
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
        kpis=state.get("csv_kpis", {}),
        weather_risk=state.get("weather_risk", {}),
        planner_draft=state.get("planner_draft", {}),
        audit_result=audit_result,
        audit_log_preview=state.get("audit_log_preview", "(none)"),
    )
    report_html = _strip_code_fences(report_html)
    deep_dive_html = _render_deep_dive_html(state)
    if deep_dive_html:
        report_html = f"{report_html}\n{deep_dive_html}"

    if not audit_result.get("passed"):
        unresolved = audit_result.get("violations", []) + audit_result.get("missing_evidence", [])
        warning_html = (
            "<h2>Audit Status</h2>"
            "<p><strong>Audit did not fully pass.</strong> Presenting a controlled failure state for human review.</p>"
            f"<ul>{''.join(f'<li>{item}</li>' for item in unresolved[:8])}</ul>"
        )
        report_html = warning_html + report_html

    return {"report_html": report_html}


def node_email(state: AppState) -> AppState:
    to_email = os.getenv("REPORT_EMAIL_TO", "").strip()
    if not to_email:
        return {}

    subject = "MSBA Ops Multi-Agent Dispatch Report"
    send_email_smtp(subject=subject, html_body=state["report_html"], to_email=to_email)
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
