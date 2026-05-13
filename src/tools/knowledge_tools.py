from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

from tools.pdf_tools import KnowledgeRag


RAG_EVAL_DATASET: list[dict[str, Any]] = [
    {
        "question": "What is the action for DQ-01 missing unique_item_id?",
        "stream": "reference",
        "expected_section": "Data Quality Rules",
        "expected_terms": ["DQ-01", "Missing `unique_item_id`", "Remove from the dispatch calculation"],
    },
    {
        "question": "How should risk_score_0_3 of 3 change travel time planning?",
        "stream": "reference",
        "expected_section": "Travel Time Buffer Policy",
        "expected_terms": ["3", "+40% buffer", "escalation"],
    },
    {
        "question": "What precipitation threshold defines heavy precipitation risk?",
        "stream": "reference",
        "expected_section": "Weather Triggers",
        "expected_terms": ["precipitation_sum", "15.0 mm/day"],
    },
    {
        "question": "Which canonical item does Heparin Na map to?",
        "stream": "reference",
        "expected_section": "Name Alias / Variant Table",
        "expected_terms": ["Heparin Na", "HEP-SOD"],
    },
    {
        "question": "What must the final dispatch report include?",
        "stream": "reference",
        "expected_section": "Reporting Requirements",
        "expected_terms": ["Weather risk summary", "Valid vs excluded shipment counts", "SLA risk flags"],
    },
]


def _clean_cell(cell: str) -> str:
    return cell.strip().replace("**", "")


def _parse_markdown_table(lines: list[str]) -> list[dict[str, str]]:
    if len(lines) < 2:
        return []

    header = [_clean_cell(cell) for cell in lines[0].strip().strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        if "|" not in line:
            continue
        cells = [_clean_cell(cell) for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def _extract_heading_blocks(text: str) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    current_heading = "Document Overview"
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("#"):
            if current_lines:
                blocks.append((current_heading, current_lines))
            current_heading = line.lstrip("#").strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        blocks.append((current_heading, current_lines))
    return blocks


def _find_first_table(block_lines: list[str]) -> list[dict[str, str]]:
    table_lines: list[str] = []
    collecting = False

    for line in block_lines:
        if "|" in line:
            collecting = True
            table_lines.append(line)
            continue
        if collecting:
            break

    return _parse_markdown_table(table_lines)


def _rows_to_mapping(rows: Iterable[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        row_key = row.get(key, "").strip()
        if row_key:
            mapping[row_key] = row
    return mapping


def load_reference_facts(markdown_path: str) -> dict[str, Any]:
    text = Path(markdown_path).read_text(encoding="utf-8")
    blocks = _extract_heading_blocks(text)

    by_heading: dict[str, list[dict[str, str]]] = {}
    for heading, lines in blocks:
        table = _find_first_table(lines)
        if table:
            by_heading[heading] = table

    canonical_items = by_heading.get("A.1 Canonical Item Master (Authoritative)", [])
    alias_matches = by_heading.get("A.2 Name Alias / Variant Table (Accepted)", [])
    legacy_id_map = by_heading.get("A.3 Legacy / Deprecated / Invalid Identifier Mapping", [])
    dq_rules = by_heading.get("11. Data Quality Rules (Anomaly Definitions)", [])
    weather_thresholds = by_heading.get("5.1 Weather Triggers (Daily Index)", [])
    buffer_policy = by_heading.get("5.2 Travel Time Buffer Policy (Score-based)", [])
    reporting_requirements = [
        line.strip("- ").strip()
        for heading, lines in blocks
        if heading == "14. Reporting Requirements"
        for line in lines
        if line.strip().startswith("-")
    ]

    canonical_by_id: dict[str, list[dict[str, str]]] = {}
    for row in canonical_items:
        item_id = row.get("item_id", "").strip()
        if item_id:
            canonical_by_id.setdefault(item_id, []).append(row)

    exact_name_map: dict[str, dict[str, dict[str, str]]] = {}
    for row in canonical_items:
        item_id = row.get("item_id", "").strip()
        canonical_name = row.get("canonical_item_name", "").strip().lower()
        if item_id and canonical_name:
            exact_name_map.setdefault(item_id, {})[canonical_name] = row
    alias_by_name = {
        row.get("alias_name", "").strip().lower(): row
        for row in alias_matches
        if row.get("alias_name", "").strip()
    }

    return {
        "canonical_items": canonical_items,
        "canonical_by_id": canonical_by_id,
        "exact_name_map": exact_name_map,
        "canonical_by_canonical_id": _rows_to_mapping(canonical_items, "canonical_item_id"),
        "alias_matches": alias_matches,
        "alias_by_name": alias_by_name,
        "legacy_id_map": legacy_id_map,
        "legacy_by_item_id": _rows_to_mapping(legacy_id_map, "legacy_item_id"),
        "dq_rules": dq_rules,
        "dq_rule_by_id": _rows_to_mapping(dq_rules, "Rule ID"),
        "weather_thresholds": weather_thresholds,
        "weather_threshold_by_condition": _rows_to_mapping(weather_thresholds, "Condition"),
        "buffer_policy": buffer_policy,
        "buffer_policy_by_score": _rows_to_mapping(buffer_policy, "risk_score_0_3"),
        "reporting_requirements": reporting_requirements,
    }


def evaluate_retrieval_results(
    eval_items: list[dict[str, Any]],
    retrieved_results: dict[str, list[Document]],
    *,
    k: int,
) -> dict[str, Any]:
    item_results: list[dict[str, Any]] = []
    hits = 0
    grounded_hits = 0

    for item in eval_items:
        docs = retrieved_results.get(item["question"], [])[:k]
        combined = "\n".join(doc.page_content for doc in docs)
        expected_section = item["expected_section"].lower()
        section_hit = any(
            expected_section in (
                f"{doc.metadata.get('section_title', '')}\n{doc.page_content}"
            ).lower()
            for doc in docs
        )
        grounded_hit = all(term.lower() in combined.lower() for term in item["expected_terms"])
        hits += int(section_hit)
        grounded_hits += int(grounded_hit)
        item_results.append(
            {
                "question": item["question"],
                "expected_section": item["expected_section"],
                "section_hit": section_hit,
                "grounded_hit": grounded_hit,
                "top_sources": [
                    {
                        "source_name": doc.metadata.get("source_name"),
                        "section_title": doc.metadata.get("section_title"),
                    }
                    for doc in docs
                ],
            }
        )

    total = max(len(eval_items), 1)
    return {
        "recall_at_k": round(hits / total, 3),
        "grounded_answer_accuracy": round(grounded_hits / total, 3),
        "k": k,
        "results": item_results,
    }


def run_rag_eval(
    rag: KnowledgeRag,
    vectordb: Any,
    *,
    eval_items: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    dataset = eval_items or RAG_EVAL_DATASET
    retrieved_results = {
        item["question"]: rag.retrieve(vectordb, item["question"], k=k, stream=item.get("stream"))
        for item in dataset
    }
    return evaluate_retrieval_results(dataset, retrieved_results, k=k)
