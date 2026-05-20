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
        "- sla_risk_flags: array of strings\n"
        "- compliance_notes: array of strings\n"
        "- cited_rules: array of strings\n\n"
        "Weather contract:\n"
        "- Use only the provided weather_risk fields.\n"
        "- Buffer policy: 0 -> 0, 1 -> 10, 2 -> 25, 3 -> 40 plus escalation.\n"
        "- If audit_feedback is present, explicitly fix the flagged issues.\n"
        "- Keep the output limited to the implemented scope: weather buffers, SLA risk, data-quality reconciliation, corridor KPIs, escalation, and monitoring.\n"
        "- Do not include vehicle-capacity, packing, staffing, driver, fleet, or resource-planning claims in any JSON field.\n"
        "- SLA tiering is corridor-level only. Do not assign stricter SLA tiers, special SLA rules, or special escalation rules by medicine, item, product class, or cold-chain status unless the retrieved policy explicitly says so.\n"
        "- Ground escalation only in weather risk score, corridor SLA tier, DQ rule codes, excluded-unit counts, and deterministic SLA risk flags.\n"
        "- Describe weather evidence factually. Avoid causal language such as saying a route score is 'driven by' one condition unless that causal rule is explicit in the provided data.\n"
        "- Do not make forward-looking claims such as a KPI pattern 'will continue'; describe only the current evidence and monitoring triggers.\n"
        "- Do not describe what a report will include; write only the recommendation itself.\n"
        "- Avoid legalistic or absolute compliance language. Use evidence-backed operational wording instead.",
    ),
    (
        "user",
        "Business context:\n{business_context}\n\n"
        "Retrieved policy context:\n{policy_context}\n\n"
        "Retrieved reference context:\n{reference_context}\n\n"
        "Trend analysis:\n{trend_analysis}\n\n"
        "Corridor KPIs:\n{corridor_kpis}\n\n"
        "Deterministic SLA risk flags:\n{sla_risk_flags}\n\n"
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
        "Corridor KPIs:\n{corridor_kpis}\n\n"
        "Deterministic SLA risk flags:\n{sla_risk_flags}\n\n"
        "KPIs:\n{kpis}\n\n"
        "Weather risk:\n{weather_risk}\n\n"
        "Audit log summary:\n{audit_log_summary}\n\n"
        "Return only strict JSON.",
    ),
])


REPORT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are ReportAgent. Produce a crisp HTML decision memo for senior operations leadership. Use headings, short "
        "paragraphs, and short bullet lists. Keep the report rule-grounded, decision-oriented, and based on exact metrics. "
        "Return raw HTML only, with no markdown code fences.",
    ),
    (
        "user",
        "Business context:\n{business_context}\n\n"
        "Ops insights:\n{ops_insights}\n\n"
        "Trend analysis:\n{trend_analysis}\n\n"
        "Corridor KPIs:\n{corridor_kpis}\n\n"
        "Deterministic SLA risk flags:\n{sla_risk_flags}\n\n"
        "KPI summary:\n{kpis}\n\n"
        "Weather risk:\n{weather_risk}\n\n"
        "Planner draft JSON:\n{planner_draft}\n\n"
        "Audit result:\n{audit_result}\n\n"
        "DQ audit log preview:\n{audit_log_preview}\n\n"
        "Write HTML using this exact section order:\n"
        "Title: Start with <h1>SeeWeeS Medical Logistics Dispatch Decision Memo</h1>\n"
        "1. Executive Decision Summary: 2 short paragraphs stating the decision, business impact, and whether human escalation is needed\n"
        "2. Decision Actions: 3-4 bullets that a senior operations leader can act on immediately\n"
        "3. SLA Watch Items: concise bullets grounded in deterministic SLA risk flags, translated into business risk language\n"
        "4. Weather Route Snapshot: compact per-waypoint table using only weather_risk.per_waypoint fields: waypoint, city, risk score, precipitation, wind gust, min temperature, and true risk flags\n"
        "5. Corridor Performance Snapshot: compact table using corridor_kpis fields; emphasize Tier, valid units, excluded units, and excluded rate\n"
        "6. Rule-Grounded Rationale: 3 short bullets tying the recommendation to DQ rules, buffer policy, reconciliation counts, and retrieved policy evidence\n"
        "7. Expected KPI Impact: 3 grounded bullets only, such as protecting Tier 1 SLA margin, keeping DQ-01 rows out of planning, and improving traceability through canonical item mapping\n"
        "8. Monitoring Triggers: concise, executive-facing bullets for what should cause re-review\n"
        "Use exact metrics where available. Do not make claims about operational-planning areas that are outside the current deterministic calculations. "
        "SLA tiering is corridor-level only. Do not imply that products, medicines, product classes, or cold-chain status receive stricter SLA tiers or special escalation rules unless that is explicit in the retrieved policy. "
        "Do not say a risk is 'driven by' a condition unless the provided data explicitly defines that causal relationship; use factual wording such as 'weather flags include'. "
        "Do not make future-looking claims such as a KPI pattern 'will continue'; say what the current data show and what should be monitored. "
        "Do not use meta phrases such as 'final report will include', 'this report will include', 'ensuring full compliance', or 'as mandated by policy'. "
        "Write the report as a finished executive decision memo, not a description of the report itself. "
        "Do not mention the internal audit loop or AuditAgent unless audit_result.passed is false. "
        "Prefer plain business language over analyst narration: say what changed, why it matters, and what leadership should do next. "
        "Keep detailed item mixes, reconciliation samples, excluded-row samples, and long tables out of the email body; those belong in the attached appendix. "
        "Do not include an appendix section in the email body.\n",
    ),
])
