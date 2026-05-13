from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from tools.knowledge_tools import evaluate_retrieval_results, load_reference_facts


SAMPLE_PLAYBOOK = """
## 11. Data Quality Rules (Anomaly Definitions)
| Rule ID | Description | Action |
|---|---|---|
| DQ-01 | Missing `unique_item_id` | Remove from the dispatch calculation |
| DQ-03 | `item_name` mismatch for valid `item_id` | Flag for investigation |

## 14. Reporting Requirements
- Weather risk summary
- Valid vs excluded shipment counts
- SLA risk flags

### 5.1 Weather Triggers (Daily Index)
| Condition | Open-Meteo Daily Variable | Threshold |
|---|---|---|
| Heavy Precipitation Risk | `precipitation_sum` | >= 15.0 mm/day |

### 5.2 Travel Time Buffer Policy (Score-based)
| risk_score_0_3 | Travel Time Adjustment |
|---:|---|
| 0 | No buffer |
| 3 | +40% buffer + escalation |

### A.1 Canonical Item Master (Authoritative)
| canonical_item_id | item_id | canonical_item_name | medicine_type | temp_control | product_class |
|---|---:|---|---|---|---|
| HEP-SOD | 10050 | Heparin Sodium | Anticoagulant | Room Temp (20-25C) | Anticoagulant |
| ALB-INH | 10070 | Albuterol Inhaler | Bronchodilator | Room Temp (20-25C) | Respiratory |

### A.2 Name Alias / Variant Table (Accepted)
| alias_name | canonical_item_id | confidence_tier | notes |
|---|---|---|---|
| Heparin Na | HEP-SOD | ALIAS_MATCH | Abbreviation |

### A.3 Legacy / Deprecated / Invalid Identifier Mapping
| legacy_item_id | canonical_item_id | rule | rationale |
|---:|---|---|---|
| 1070 | ALB-INH | LEGACY_ID_MAP | Truncated ID found in older CSV exports |
""".strip()


def test_load_reference_facts_parses_alias_and_legacy_tables(tmp_path: Path):
    playbook_path = tmp_path / "playbook.md"
    playbook_path.write_text(SAMPLE_PLAYBOOK, encoding="utf-8")

    reference_facts = load_reference_facts(str(playbook_path))

    assert reference_facts["alias_by_name"]["heparin na"]["canonical_item_id"] == "HEP-SOD"
    assert reference_facts["legacy_by_item_id"]["1070"]["canonical_item_id"] == "ALB-INH"
    assert reference_facts["buffer_policy_by_score"]["3"]["Travel Time Adjustment"] == "+40% buffer + escalation"


def test_evaluate_retrieval_results_scores_hits():
    eval_items = [
        {
            "question": "Which item does Heparin Na map to?",
            "expected_section": "Name Alias / Variant Table",
            "expected_terms": ["Heparin Na", "HEP-SOD"],
        }
    ]
    retrieved_results = {
        "Which item does Heparin Na map to?": [
            Document(
                page_content="A.2 Name Alias / Variant Table\n\nHeparin Na maps to HEP-SOD.",
                metadata={"section_title": "A.2 Name Alias / Variant Table", "source_name": "playbook.md"},
            )
        ]
    }

    scores = evaluate_retrieval_results(eval_items, retrieved_results, k=1)

    assert scores["recall_at_k"] == 1.0
    assert scores["grounded_answer_accuracy"] == 1.0
