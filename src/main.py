from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must be before importing graph/agents

from tracing import init_langsmith_tracing

init_langsmith_tracing()  # must be before importing graph/agents

from graph import build_graph


def _default_knowledge_sources() -> list[str]:
    candidates = [
        "data/SeeWeeS Specialty Dispatch Playbook.pdf",
        "data-for-enhancement/SeeWeeS Specialty Dispatch Playbook.md",
        "data/About SeeWeeS Specialty distribution.pdf",
    ]
    return [path for path in candidates if Path(path).exists()]


if __name__ == "__main__":
    app = build_graph()

    state = {
        "pdf_path": "data/SeeWeeS Specialty Dispatch Playbook.pdf",
        "csv_path": "data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv",
        "reference_markdown_path": "data-for-enhancement/SeeWeeS Specialty Dispatch Playbook.md",
        "knowledge_sources": _default_knowledge_sources(),
        "max_audit_retries": int(os.getenv("AUDIT_MAX_RETRIES", "2")),
    }

    final = app.invoke(state)

    print("\n=== RAG EVAL ===\n")
    print(final.get("rag_eval_results", {}))

    print("\n=== AUDIT RESULT ===\n")
    print(final.get("audit_result", {}))

    report_html = final.get("report_html", "")
    print("\n=== REPORT (first 2500 chars) ===\n")
    print(report_html[:2500])
