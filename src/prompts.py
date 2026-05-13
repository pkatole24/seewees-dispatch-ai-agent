from langchain_core.prompts import ChatPromptTemplate


KNOWLEDGE_CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are KnowledgeAgent. Synthesize retrieved operational policy and appendix evidence into a concise "
        "brief for downstream agents. Focus on rules, KPI definitions, dispatch constraints, reporting requirements, "
        "and any exception-handling logic.",
    ),
    (
        "user",
        "Policy evidence:\n{policy_context}\n\nReference evidence:\n{reference_context}\n\n"
        "Return:\n"
        "1) Core business rules\n"
        "2) KPI and reporting expectations\n"
        "3) Data-quality rules and exception handling\n"
        "4) Dispatch or weather guardrails\n",
    ),
])


OPS_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are OpsTrendAgent. Turn deterministic trend outputs into leadership-ready insight. "
        "Do not invent metrics beyond the provided data. Surface the most important quantitative findings, "
        "including day-over-day shifts, DQ reason counts, correction patterns, and item spikes when present.",
    ),
    (
        "user",
        "Summary:\n{summary}\n\n"
        "KPIs:\n{kpis}\n\n"
        "Trend analysis:\n{trend_analysis}\n\n"
        "Audit-log highlights:\n{audit_log_highlights}\n\n"
        "Anomalies:\n{anomalies_md}\n\n"
        "Return:\n"
        "- Key operational changes vs history\n"
        "- Data quality business impact\n"
        "- Deep-dive quantitative findings\n"
        "- Immediate actions\n"
        "- What leadership should monitor next\n",
    ),
])


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are PlannerAgent. Produce a compliant dispatch recommendation grounded in retrieved evidence and trend "
        "outputs. You must return strict JSON only.\n\n"
        "Required JSON keys:\n"
        "- executive_summary: string\n"
        "- recommended_buffer_pct: integer\n"
        "- escalation_required: boolean\n"
        "- dispatch_actions: array of strings\n"
        "- monitoring_points: array of strings\n"
        "- contingency_triggers: array of strings\n"
        "- expected_kpi_impacts: array of strings\n"
        "- compliance_notes: array of strings\n"
        "- cited_rules: array of strings\n\n"
        "Weather contract:\n"
        "- Use only the provided weather_risk fields.\n"
        "- Buffer policy: 0 -> 0, 1 -> 10, 2 -> 25, 3 -> 40 plus escalation.\n"
        "- If audit_feedback is present, explicitly fix the flagged issues.",
    ),
    (
        "user",
        "Business context:\n{business_context}\n\n"
        "Retrieved policy context:\n{policy_context}\n\n"
        "Retrieved reference context:\n{reference_context}\n\n"
        "Trend analysis:\n{trend_analysis}\n\n"
        "KPIs:\n{kpis}\n\n"
        "Ops insights:\n{ops_insights}\n\n"
        "Weather risk:\n{weather_risk}\n\n"
        "Audit feedback:\n{audit_feedback}\n\n"
        "Return only strict JSON.",
    ),
])


AUDIT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are AuditAgent. Review the planner draft for rule compliance, evidence support, and consistency with the "
        "provided data. Return strict JSON only.\n\n"
        "Required JSON keys:\n"
        "- passed: boolean\n"
        "- violations: array of strings\n"
        "- unsupported_claims: array of strings\n"
        "- missing_evidence: array of strings\n"
        "- revision_instructions: array of strings\n"
        "- compliance_summary: string\n"
    ),
    (
        "user",
        "Planner draft JSON:\n{planner_draft}\n\n"
        "Business context:\n{business_context}\n\n"
        "Retrieved policy context:\n{policy_context}\n\n"
        "Retrieved reference context:\n{reference_context}\n\n"
        "Trend analysis:\n{trend_analysis}\n\n"
        "KPIs:\n{kpis}\n\n"
        "Weather risk:\n{weather_risk}\n\n"
        "Audit log summary:\n{audit_log_summary}\n\n"
        "Return only strict JSON.",
    ),
])


REPORT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are ReportAgent. Produce a crisp HTML report for leadership. Use headings, short paragraphs, and bullet "
        "lists. Keep the report audit-backed and decision-oriented. Return raw HTML only, with no markdown code fences. "
        "Include concrete quantities from the provided trend analysis rather than abstract summaries.",
    ),
    (
        "user",
        "Business context:\n{business_context}\n\n"
        "Ops insights:\n{ops_insights}\n\n"
        "Trend analysis:\n{trend_analysis}\n\n"
        "KPI summary:\n{kpis}\n\n"
        "Weather risk:\n{weather_risk}\n\n"
        "Planner draft JSON:\n{planner_draft}\n\n"
        "Audit result:\n{audit_result}\n\n"
        "DQ audit log preview:\n{audit_log_preview}\n\n"
        "Write HTML with sections for:\n"
        "- Executive summary\n"
        "- Trend shifts and data quality impact\n"
        "- Dispatch recommendation\n"
        "- Audit-backed compliance notes\n"
        "- What to monitor next\n"
        "Use exact metrics where available.\n",
    ),
])
