from __future__ import annotations

from graph import (
    _build_sla_risk_flags,
    _filter_implemented_scope_context,
    _ensure_weather_snapshot_section,
    _render_deep_dive_html,
    _render_deep_dive_text,
    _render_weather_snapshot_html,
    _sanitize_planner_output,
    _strip_code_fences,
    _weather_unavailable_risk,
    apply_deterministic_audit_checks,
    node_report,
    route_after_audit,
)


def test_apply_deterministic_audit_checks_rejects_wrong_buffer_and_missing_citations():
    state = {
        "weather_risk": {"risk_score_0_3": 3},
        "planner_draft": {
            "recommended_buffer_pct": 25,
            "escalation_required": False,
            "cited_rules": [],
        },
    }
    audit_result = {
        "passed": True,
        "violations": [],
        "missing_evidence": [],
        "revision_instructions": [],
    }

    checked = apply_deterministic_audit_checks(state, audit_result)

    assert checked["passed"] is False
    assert any("required buffer 40" in item for item in checked["violations"])
    assert any("cited_rules" in item for item in checked["revision_instructions"])


def test_apply_deterministic_audit_checks_rejects_unsupported_capacity_claims():
    state = {
        "weather_risk": {"risk_score_0_3": 1},
        "planner_draft": {
            "recommended_buffer_pct": 10,
            "escalation_required": False,
            "cited_rules": ["Travel Time Buffer Policy"],
            "sla_risk_flags": ["Tier 1 monitor"],
            "dispatch_actions": ["Truck capacity compliance is respected."],
        },
        "sla_risk_flags": ["Tier 1 monitor"],
    }
    audit_result = {
        "passed": True,
        "violations": [],
        "missing_evidence": [],
        "revision_instructions": [],
    }

    checked = apply_deterministic_audit_checks(state, audit_result)

    assert checked["passed"] is False
    assert any("internal reasoning or meta-reporting language" in item for item in checked["violations"])


def test_sanitize_planner_output_removes_out_of_scope_language_before_audit():
    planner_draft = {
        "dispatch_actions": [
            "Truck capacity compliance is respected and packing constraint checks are complete.",
            "The final report will include SLA flags, ensuring full compliance.",
            "Stricter SLA tiering is driven by medicine-type-specific escalation and will continue.",
        ],
        "monitoring_points": ["Watch capacity use by corridor."],
    }

    sanitized = _sanitize_planner_output(planner_draft)
    checked = apply_deterministic_audit_checks(
        {
            "weather_risk": {"risk_score_0_3": 0},
            "planner_draft": {
                **sanitized,
                "recommended_buffer_pct": 0,
                "escalation_required": False,
                "cited_rules": ["Travel Time Buffer Policy"],
            },
        },
        {
            "passed": True,
            "violations": [],
            "missing_evidence": [],
            "revision_instructions": [],
        },
    )

    sanitized_text = str(sanitized).lower()
    assert "truck capacity" not in sanitized_text
    assert "packing constraint" not in sanitized_text
    assert "final report will include" not in sanitized_text
    assert "ensuring full compliance" not in sanitized_text
    assert "stricter sla tiering" not in sanitized_text
    assert "driven by" not in sanitized_text
    assert "medicine-type-specific escalation" not in sanitized_text
    assert checked["passed"] is True


def test_build_sla_risk_flags_adds_weather_buffer_for_tier_one_corridor():
    flags = _build_sla_risk_flags(
        {
            "trend_analysis": {
                "corridor_kpis": [
                    {
                        "corridor_id": "C1_I95_NJ_BOS",
                        "default_sla_tier": "Tier 1",
                        "planning_window_valid_units": 16,
                    }
                ]
            },
            "sla_risk_flags": [],
        },
        {"risk_score_0_3": 2},
    )

    assert any("25% travel buffer" in flag for flag in flags)


def test_weather_unavailable_risk_is_safe_default():
    risk = _weather_unavailable_risk("bad gateway")

    assert risk["risk_score_0_3"] == 0
    assert risk["weather_data_status"] == "unavailable"
    assert risk["risk_flags"]["weather_data_unavailable"] is True


def test_filter_implemented_scope_context_removes_resource_sections_only():
    context = """
## 8. Truck Capacity & Packing Model
Do not pass this to planner.

## 11. Data Quality Rules (Anomaly Definitions)
DQ-01 Missing unique_item_id

## 13. Resource Constraints and Allocation Policy
Do not pass this either.

## Appendix A - Item Master Appendix
Alias and legacy mappings stay in scope.
""".strip()

    filtered = _filter_implemented_scope_context(context)

    assert "Truck Capacity" not in filtered
    assert "Resource Constraints" not in filtered
    assert "DQ-01" in filtered
    assert "Alias and legacy mappings" in filtered


def test_route_after_audit_loops_until_retry_budget_is_exhausted():
    assert route_after_audit(
        {
            "audit_result": {"passed": False},
            "planner_attempts": 1,
            "max_audit_retries": 2,
        }
    ) == "planner"

    assert route_after_audit(
        {
            "audit_result": {"passed": False},
            "planner_attempts": 2,
            "max_audit_retries": 2,
        }
    ) == "report"

    assert route_after_audit(
        {
            "audit_result": {"passed": True},
            "planner_attempts": 1,
            "max_audit_retries": 2,
        }
    ) == "report"


def test_report_helpers_strip_fences_and_render_deep_dive_tables():
    fenced = "```html\n<h1>Hello</h1>\n```"
    assert _strip_code_fences(fenced) == "<h1>Hello</h1>"

    html = _render_deep_dive_html(
        state := {
            "trend_analysis": {
                "deep_dive_tables": {
                    "daily_valid_units": [
                        {"shipment_date": "2026-03-06", "planning_day": "Day0", "valid_units": 10}
                    ],
                    "corridor_day_breakdown": [],
                    "item_spikes": [],
                    "correction_breakdown": [],
                    "exclusion_breakdown": [],
                    "daily_excluded_units": [],
                    "corrected_samples": [],
                    "unresolved_samples": [],
                }
            }
        }
    )
    assert "Deep-Dive Analytics Appendix" in html
    assert "Daily Valid Shipment Trend" in html

    text = _render_deep_dive_text(state)
    assert "MSBA Ops Deep-Dive Analytics Appendix" in text
    assert "shipment_date | planning_day | valid_units" in text


def test_weather_snapshot_section_uses_per_waypoint_weather_evidence():
    weather_risk = {
        "worst_waypoint": {"waypoint": "W2"},
        "per_waypoint": [
            {
                "waypoint": "W1",
                "city": "Newark, NJ",
                "risk_score_0_3": 0,
                "max_precip_mm_day": 1.1,
                "max_wind_gust_kmh": 30.2,
                "min_temp_c": 4.4,
                "risk_flags": {"heavy_rain_risk": False, "high_wind_risk": False},
            },
            {
                "waypoint": "W2",
                "city": "Bronx, NY",
                "risk_score_0_3": 2,
                "max_precip_mm_day": 21.2,
                "max_wind_gust_kmh": 48.6,
                "min_temp_c": 2.1,
                "risk_flags": {"heavy_rain_risk": True, "high_wind_risk": True},
            },
        ],
    }

    weather_html = _render_weather_snapshot_html(weather_risk)
    report_html = _ensure_weather_snapshot_section(
        "<h2>5. Corridor Performance Snapshot</h2><p>Corridor table.</p>",
        weather_risk,
    )

    assert "Weather Route Snapshot" in weather_html
    assert "W2 (worst)" in weather_html
    assert "heavy_rain_risk, high_wind_risk" in weather_html
    assert report_html.index("Weather Route Snapshot") < report_html.index("Corridor Performance Snapshot")


def test_node_report_appends_review_status_after_memo(monkeypatch):
    def fake_report_agent(**_kwargs):
        return "<h1>SeeWeeS Medical Logistics Dispatch Decision Memo</h1><p>Decision first.</p>"

    monkeypatch.setattr("graph.run_report_agent", fake_report_agent)

    result = node_report(
        {
            "audit_result": {
                "passed": False,
                "violations": ["Needs confirmation."],
                "missing_evidence": [],
            },
            "trend_analysis": {},
        }
    )

    report_html = result["report_html"]
    assert report_html.index("SeeWeeS Medical Logistics Dispatch Decision Memo") < report_html.index("Review Status")
    assert "Human review required before dispatch approval" in report_html
