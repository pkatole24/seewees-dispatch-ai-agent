from __future__ import annotations

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from prompts import (
    AUDIT_PROMPT,
    KNOWLEDGE_CONTEXT_PROMPT,
    OPS_ANALYSIS_PROMPT,
    PLANNER_PROMPT,
    REPORT_PROMPT,
)

_LLM: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _LLM
    if _LLM is None:
        _LLM = ChatOpenAI(
            model="gpt-5.1",
            temperature=0.2,
            tags=["msba-demo", "multi-agent"],
            metadata={"repo": "MSBA_AI_Agents_Demo"},
        )
    return _LLM


def _extract_json_block(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def run_context_agent(policy_context: str, reference_context: str) -> str:
    return _get_llm().invoke(
        KNOWLEDGE_CONTEXT_PROMPT.format_messages(
            policy_context=policy_context,
            reference_context=reference_context,
        )
    ).content


def run_ops_agent(
    summary: Dict[str, Any],
    kpis: Dict[str, Any],
    trend_analysis: Dict[str, Any],
    audit_log_highlights: str,
    anomalies_md: str,
) -> str:
    return _get_llm().invoke(
        OPS_ANALYSIS_PROMPT.format_messages(
            summary=summary,
            kpis=kpis,
            trend_analysis=trend_analysis,
            audit_log_highlights=audit_log_highlights,
            anomalies_md=anomalies_md,
        )
    ).content


def run_planner_agent(
    business_context: str,
    policy_context: str,
    reference_context: str,
    trend_analysis: Dict[str, Any],
    corridor_kpis: list[Dict[str, Any]],
    sla_risk_flags: list[str],
    kpis: Dict[str, Any],
    ops_insights: str,
    weather_risk: Dict[str, Any],
    audit_feedback: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    response = _get_llm().invoke(
        PLANNER_PROMPT.format_messages(
            business_context=business_context,
            policy_context=policy_context,
            reference_context=reference_context,
            trend_analysis=trend_analysis,
            corridor_kpis=corridor_kpis,
            sla_risk_flags=sla_risk_flags,
            kpis=kpis,
            ops_insights=ops_insights,
            weather_risk=weather_risk,
            audit_feedback=audit_feedback or {},
        )
    ).content
    return _extract_json_block(response)


def run_audit_agent(
    planner_draft: Dict[str, Any],
    business_context: str,
    policy_context: str,
    reference_context: str,
    trend_analysis: Dict[str, Any],
    corridor_kpis: list[Dict[str, Any]],
    sla_risk_flags: list[str],
    kpis: Dict[str, Any],
    weather_risk: Dict[str, Any],
    audit_log_summary: Dict[str, Any],
) -> Dict[str, Any]:
    response = _get_llm().invoke(
        AUDIT_PROMPT.format_messages(
            planner_draft=planner_draft,
            business_context=business_context,
            policy_context=policy_context,
            reference_context=reference_context,
            trend_analysis=trend_analysis,
            corridor_kpis=corridor_kpis,
            sla_risk_flags=sla_risk_flags,
            kpis=kpis,
            weather_risk=weather_risk,
            audit_log_summary=audit_log_summary,
        )
    ).content
    return _extract_json_block(response)


def run_report_agent(
    business_context: str,
    ops_insights: str,
    trend_analysis: Dict[str, Any],
    corridor_kpis: list[Dict[str, Any]],
    sla_risk_flags: list[str],
    kpis: Dict[str, Any],
    weather_risk: Dict[str, Any],
    planner_draft: Dict[str, Any],
    audit_result: Dict[str, Any],
    audit_log_preview: str,
) -> str:
    return _get_llm().invoke(
        REPORT_PROMPT.format_messages(
            business_context=business_context,
            ops_insights=ops_insights,
            trend_analysis=trend_analysis,
            corridor_kpis=corridor_kpis,
            sla_risk_flags=sla_risk_flags,
            kpis=kpis,
            weather_risk=weather_risk,
            planner_draft=planner_draft,
            audit_result=audit_result,
            audit_log_preview=audit_log_preview,
        )
    ).content
