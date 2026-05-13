from __future__ import annotations

from graph import _render_deep_dive_html, _strip_code_fences, apply_deterministic_audit_checks, route_after_audit


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
        {
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
