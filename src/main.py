from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must be before importing graph/agents

from tracing import init_langsmith_tracing

init_langsmith_tracing()  # must be before importing graph/agents

from graph import build_graph


STAGE_LABELS = {
    "knowledge": "Retrieve policy and reference context",
    "trend_analysis": "Analyze shipments and DQ trends",
    "weather": "Fetch route weather and buffer risk",
    "planner": "Draft dispatch recommendation",
    "audit": "Audit recommendation against rules",
    "report": "Generate executive report",
    "email": "Send email report",
}


def _default_knowledge_sources() -> list[str]:
    candidates = [
        "data/SeeWeeS Specialty Dispatch Playbook.pdf",
        "data-for-enhancement/SeeWeeS Specialty Dispatch Playbook.md",
        "data/About SeeWeeS Specialty distribution.pdf",
    ]
    return [path for path in candidates if Path(path).exists()]


def _progress_bar(completed: int, total: int, width: int = 24) -> str:
    filled = min(width, round(width * completed / max(total, 1)))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {completed}/{total}"


def _run_graph_with_progress(app, state: dict) -> dict:
    total = len(STAGE_LABELS)
    completed_nodes: set[str] = set()
    final_state = dict(state)

    print("\nStarting MSBA multi-agent dispatch graph...\n", flush=True)
    for update in app.stream(state):
        for node_name, node_update in update.items():
            if isinstance(node_update, dict):
                final_state.update(node_update)

            completed_nodes.add(node_name)
            label = STAGE_LABELS.get(node_name, node_name)
            attempts = ""
            if node_name == "planner":
                attempts = f" (attempt {final_state.get('planner_attempts', 1)})"
            elif node_name == "audit":
                audit_status = "passed" if final_state.get("audit_result", {}).get("passed") else "needs retry/review"
                attempts = f" ({audit_status})"

            print(f"{_progress_bar(len(completed_nodes), total)} {label}{attempts}", flush=True)

    print("\nGraph run complete.\n", flush=True)
    return final_state


if __name__ == "__main__":
    app = build_graph()

    state = {
        "pdf_path": "data/SeeWeeS Specialty Dispatch Playbook.pdf",
        "csv_path": "data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv",
        "reference_markdown_path": "data-for-enhancement/SeeWeeS Specialty Dispatch Playbook.md",
        "knowledge_sources": _default_knowledge_sources(),
        "max_audit_retries": int(os.getenv("AUDIT_MAX_RETRIES", "2")),
    }

    final = _run_graph_with_progress(app, state)

    print("\n=== RAG EVAL ===\n")
    print(final.get("rag_eval_results", {}))

    print("\n=== AUDIT RESULT ===\n")
    print(final.get("audit_result", {}))

    report_html = final.get("report_html", "")
    print("\n=== REPORT (first 2500 chars) ===\n")
    print(report_html[:2500])
