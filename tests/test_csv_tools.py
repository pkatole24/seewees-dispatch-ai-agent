from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.csv_tools import analyze_csv
from tools.knowledge_tools import load_reference_facts


SAMPLE_PLAYBOOK = """
## 11. Data Quality Rules (Anomaly Definitions)
| Rule ID | Description | Action |
|---|---|---|
| DQ-01 | Missing `unique_item_id` | Remove from the dispatch calculation |
| DQ-02 | `item_id` not in master table | Flag for investigation |
| DQ-03 | `item_name` mismatch for valid `item_id` | Flag for investigation |
| DQ-04 | Duplicate `unique_item_id` | Flag for investigation |

### A.1 Canonical Item Master (Authoritative)
| canonical_item_id | item_id | canonical_item_name | medicine_type | temp_control | product_class |
|---|---:|---|---|---|---|
| RMD-100 | 10021 | Remdesivir 100mg | Antiviral | Cold (2-8C) | Antiviral |
| RMD-200 | 10021 | Remdesivir 200mg | Antiviral | Cold (2-8C) | Antiviral |
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


def test_analyze_csv_reconciles_aliases_and_legacy_ids(tmp_path: Path):
    playbook_path = tmp_path / "playbook.md"
    playbook_path.write_text(SAMPLE_PLAYBOOK, encoding="utf-8")
    reference_facts = load_reference_facts(str(playbook_path))

    csv_path = tmp_path / "shipments.csv"
    pd.DataFrame(
        [
            {
                "shipment_date": "2026-03-06",
                "planning_day": "Day0",
                "is_planning_window": 1,
                "corridor_id": "C1",
                "item_id": 10050,
                "item_name": "Heparin Na",
                "unique_item_id": "HEP-1",
                "dispatch_location": "Boston-MGH",
            },
            {
                "shipment_date": "2026-03-06",
                "planning_day": "Day0",
                "is_planning_window": 1,
                "corridor_id": "C1",
                "item_id": 1070,
                "item_name": "Albuterol Inhaler",
                "unique_item_id": "ALB-1",
                "dispatch_location": "Boston-MGH",
            },
            {
                "shipment_date": "2026-03-06",
                "planning_day": "Day0",
                "is_planning_window": 1,
                "corridor_id": "C1",
                "item_id": 10021,
                "item_name": "Remdesivir 200mg",
                "unique_item_id": None,
                "dispatch_location": "Boston-MGH",
            },
            {
                "shipment_date": "2026-03-01",
                "planning_day": "History",
                "is_planning_window": 0,
                "corridor_id": "C1",
                "item_id": 10021,
                "item_name": "Remdesivir 100mg",
                "unique_item_id": "RMD-1",
                "dispatch_location": "Boston-MGH",
            },
            {
                "shipment_date": "2026-03-02",
                "planning_day": "History",
                "is_planning_window": 0,
                "corridor_id": "C1",
                "item_id": 10021,
                "item_name": "Remdesivir Unknown",
                "unique_item_id": "RMD-2",
                "dispatch_location": "Boston-MGH",
            },
        ]
    ).to_csv(csv_path, index=False)

    result = analyze_csv(str(csv_path), reference_facts=reference_facts)

    assert result.kpis["valid_units"] == 3
    assert result.kpis["excluded_units"] == 2
    assert result.kpis["corrected_units"] == 2
    assert result.kpis["planning_window_valid_units"] == 2
    assert result.kpis["history_valid_units"] == 1

    valid_rows = result.valid_shipments.set_index("unique_item_id")
    assert valid_rows.loc["HEP-1", "canonical_item_id"] == "HEP-SOD"
    assert valid_rows.loc["HEP-1", "resolution_code"] == "alias_match"
    assert valid_rows.loc["ALB-1", "canonical_item_id"] == "ALB-INH"
    assert valid_rows.loc["ALB-1", "resolution_code"] == "legacy_id_map"

    excluded_reasons = set(result.excluded_shipments["reason_code"])
    assert "excluded_missing_unique_item_id" in excluded_reasons
    assert "excluded_unresolved" in excluded_reasons
    assert result.trend_analysis["deep_dive_tables"]["item_spikes"]
    assert result.trend_analysis["deep_dive_tables"]["correction_breakdown"]
