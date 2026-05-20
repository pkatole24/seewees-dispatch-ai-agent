# Technical & Business Documentation

## Executive Summary

SeeWeeS Specialty Distribution operates time-critical medicine dispatch from New Jersey distribution centers to hospital regions. The original prototype produced a single-pass operations report, but the challenge problem asks for a more realistic multi-agent system that can handle messy data, changing operating risk, and guardrails before recommendations reach leadership.

This project implements two core enhancements:
- `Idea 1: Self-Correction & Quality Assurance`
- `Idea 3: Deep-Dive Trend Analysis`

The resulting LangGraph workflow retrieves business rules, reconciles dirty shipment data, computes grounded operational metrics, drafts a dispatch recommendation, audits the recommendation, and only then produces an executive-facing HTML report.

The executive report is intentionally concise. Detailed reconciliation and trend tables are generated as a separate text appendix so leadership receives the decision story first while evaluators can still inspect the underlying deterministic evidence.

## Stakeholder And Pain Point

The primary stakeholder is a SeeWeeS operations leader responsible for dispatch planning and service reliability. Their pain point is that high-stakes medicine dispatch decisions depend on incomplete shipment feeds, weather risk, item-master inconsistencies, and SLA-sensitive corridors.

The system helps by:
- standardizing shipment records before analysis
- excluding invalid units with auditable reason codes
- surfacing trend shifts and corridor concentration
- applying weather-buffer policy consistently
- preventing unsupported planner recommendations from being sent as final guidance

## Key Assumptions

- Each row in the shipment feed represents one physical shipment unit.
- `unique_item_id` is required for dispatch calculation and traceability.
- The enhancement markdown playbook is the authoritative source for appendix tables, data-quality rules, corridor defaults, and reporting expectations.
- Corridor default SLA tiers are taken from the corridor catalog.
- `C1_I95_NJ_BOS` is Tier 1 and `C2_NJ_PHL` is Tier 2.
- Weather risk is scored from Open-Meteo daily aggregates using precipitation, wind gust, and minimum temperature thresholds.
- Resource allocation is not claimed as completed in this version. The included resource file is treated as future-work input.

## Data Augmentation Strategy

The project uses the original company/playbook PDFs plus the enhancement artifacts in `data-for-enhancement/`.

The main augmented data source is `Incoming_shipments_14d_multi_corridor.csv`, which adds:
- a 14-day historical and planning-window shipment feed
- two dispatch corridors
- intentional data-quality issues
- item-name variants and legacy identifiers
- Day0 and Day1 planning-window flags

The markdown playbook adds machine-readable policy and appendix tables for:
- corridor catalog and SLA tiers
- weather risk thresholds
- travel-time buffer policy
- data-quality rules
- canonical item master
- alias and legacy-ID mappings
- reporting requirements

## KPI Definitions

The system computes deterministic KPIs before the LLM report step.

- `valid_units`: rows included after reconciliation and DQ checks
- `excluded_units`: rows excluded from dispatch planning
- `excluded_rate`: excluded units divided by total rows
- `corrected_units`: rows resolved through alias or legacy-ID mapping
- `planning_window_valid_units`: valid Day0 and Day1 units
- `history_avg_daily_valid_units`: average valid historical daily units
- `planning_window_avg_daily_valid_units`: average valid units per planning-window day
- `corridor_kpis`: planning-window valid units, excluded units, excluded rate, and default SLA tier by corridor
- `sla_risk_flags`: deterministic flags for Tier 1 exposure, DQ risk, and weather-buffer risk

In the current enhancement dataset, deterministic analysis reports:
- `129` total rows
- `124` valid units
- `5` excluded units
- `3.9%` excluded rate
- `29` alias or legacy corrections
- `30` valid planning-window units
- `15.0` planning-window average daily valid units
- `7.83` historical average daily valid units

## Technical Methodology

The system is implemented as a LangGraph state machine with specialized nodes:

- `Knowledge Agent`: retrieves policy and appendix evidence from a multi-source vector store
- `Ops / Trend Agent`: converts deterministic data outputs into operational insights
- `Planner Agent`: drafts a structured dispatch recommendation as JSON
- `Audit Agent`: checks the planner draft for rule compliance and evidence support
- `Report Agent`: writes the final executive HTML report
- `Email Step`: optionally sends the report and deep-dive appendix through SMTP

The graph includes a conditional edge from audit back to planner. If the audit fails and retry budget remains, the planner receives audit feedback and regenerates the recommendation.

## Architectural Enhancements

The original single-pass workflow was expanded in four ways:

- Multi-source RAG indexes the original PDF, enhancement markdown playbook, and supporting company PDF.
- Reference extraction parses structured appendix facts for deterministic reconciliation.
- CSV analysis now performs rule-based DQ handling and trend analysis before the LLM step.
- Audit logic performs both LLM-based review and deterministic checks for critical policies.

The deterministic audit currently checks:
- weather risk score to travel-buffer mapping
- escalation requirement when risk score is `3`
- required rule citations
- required SLA risk flags
- prevention of unsupported truck-capacity or resource-allocation claims

## Agent Design

The agents are intentionally separated by responsibility.

- `KnowledgeAgent` summarizes retrieved rules, KPI expectations, exception handling, and dispatch guardrails.
- `OpsTrendAgent` explains trend changes, DQ impact, correction patterns, and item spikes from deterministic outputs.
- `PlannerAgent` returns strict JSON so downstream audit checks can inspect the recommendation.
- `AuditAgent` reviews policy compliance, unsupported claims, missing evidence, and revision instructions.
- `ReportAgent` produces leadership-ready HTML using exact metrics and audit-backed recommendation content.

This design makes the system more agentic than a linear chain because intermediate recommendations can be rejected and corrected before final reporting.

## Results And Business Insights

The current sample report surfaces several useful SeeWeeS operations findings:

- Planning-window valid shipment volume increased to `15.0` units per day versus a historical average of `7.83`.
- Planning-window demand is split across two corridors: `16` valid units for the Tier 1 Boston corridor and `14` valid units for the Tier 2 Philadelphia corridor.
- Data quality remains a business risk: `5` rows are excluded due to missing `unique_item_id`.
- Alias and legacy-ID reconciliation preserve usable records that would otherwise be noisy or misleading.
- Weather buffer policy is applied from the playbook risk-score table.
- A waypoint-level weather route snapshot shows the evidence behind the buffer recommendation.
- SLA flags now distinguish Tier 1 exposure from general shipment volume.

These outputs help leadership understand not only what changed, but why a particular dispatch recommendation is safer than an unaudited single-pass answer.

The attached deep-dive appendix provides the analyst-facing tables behind the executive report:
- daily valid shipment trend
- corridor-by-day breakdown
- item spike analysis
- correction and exclusion breakdowns
- sample corrected rows
- sample excluded or unresolved rows

## Validation Strategy

Validation combines deterministic tests and runtime reporting.

The test suite validates:
- appendix parsing for alias, legacy-ID, and corridor tables
- shipment reconciliation for alias and legacy records
- exclusion behavior for missing and unresolved identifiers
- corridor KPI and SLA tier assignment
- audit-loop routing behavior
- deterministic rejection of wrong weather buffers
- deterministic rejection of unsupported truck-capacity/resource claims
- report helper formatting and weather snapshot rendering
- retrieval-evaluation scoring helpers

Current local test result:

```text
14 passed
```

The RAG layer also includes a small built-in evaluation set that checks retrieval coverage for:
- DQ-01 handling
- weather-buffer policy
- weather trigger thresholds
- alias mapping
- reporting requirements

## Limitations And Next Steps

The current project intentionally focuses on Ideas 1 and 3.

Limitations:
- Full resource allocation is not implemented.
- `Resource_availability_48h.csv` is included but not used in the current report flow.
- Weather output is route-level rather than a complete corridor-by-day weather matrix.
- SLA risk flags are deterministic and policy-grounded, but they do not estimate exact travel time or actual SLA breach probability.

Next steps:
- Add resource allocation only if the team chooses to expand into Idea 5.
- Add deterministic truck-capacity math before making capacity-compliance claims.
- Add corridor/day weather scoring for each corridor.
- Add richer report rendering or screenshots for the final presentation.
- Export the technical/business documentation to PDF if required by the submission format.
