# Data Quality Framework

## Philosophy

This pipeline follows a **flag-not-drop** philosophy. When a record is suspicious or definitively wrong, a boolean column is added to mark it rather than silently removing the row from silver. This serves three purposes:

1. **Transparency:** analysts see the full population, including dirty records, and can measure the scale of each quality problem.
2. **Investigation:** operations teams can filter to flagged rows and trace issues to their source (mis-keyed IDs, system retransmissions, integration failures).
3. **Auditability:** downstream consumers can decide whether to include or exclude flagged rows based on their use case — a dashboards team may exclude `invalid_arrival_sequence_flag` rows from delay averages, while an engineering team investigating the flag pattern should include them.

The only exceptions to flag-not-drop are: (1) rows quarantined at bronze ingestion for null primary keys (a row with no `vessel_call_id` is structurally unusable), and (2) negative `actual_duration_minutes` in container moves (replaced with NULL rather than kept as a negative value, since downstream averages would be silently wrong).

---

## Quality Checks (14 total)

### Silver vessel calls (6 checks)

| Check | Rule | Status thresholds |
|---|---|---|
| **Duplicate vessel call IDs** | Silver dedup must leave exactly one row per `vessel_call_id`. Any duplicate means the deduplication step failed. | > 0 rows → FAIL |
| **Missing ETA rate** | ETA is required to schedule operations. Rates above 5% suggest a systemic upstream extraction problem, not isolated data entry errors. | > 5% → WARN, > 15% → FAIL |
| **Missing ATA rate** | ATA is null for in-progress calls — expected. High rates across a historical table indicate recording failures. | > 10% → WARN, > 25% → FAIL |
| **Invalid arrival sequence** | `ATA > ATD`: a vessel cannot depart before it arrives. Each occurrence is a data error. | > 0 → WARN |
| **Invalid departure sequence** | `ETD < ETA`: planned departure before planned arrival is a scheduling entry error. | > 0 → WARN |
| **Large delay outliers** | Delay exceeds the Tukey upper fence (`Q3 + 1.5×IQR`, minimum 24 hours). A small number of genuine extreme delays is expected; a high rate indicates systematic mis-recording. | > 1% → WARN, > 5% → FAIL |

### Silver crane assignments (3 checks)

| Check | Rule | Status thresholds |
|---|---|---|
| **End before start** | `actual_end < actual_start` is physically impossible. Rates above 2% suggest a systematic timestamp extraction or timezone error. | > 2% → WARN, > 10% → FAIL |
| **Crane overlaps** | A physical crane cannot serve two calls simultaneously. Overlapping assignments arise from scheduling conflicts or data entry errors. Both parties in each conflicting pair are flagged. | > 5% → WARN, > 15% → FAIL |
| **Orphan assignments** | Every `vessel_call_id` in crane assignments must exist in silver vessel calls. Orphans arise when a call is quarantined without cascading removal to its associated tables. | > 0 → FAIL |

### Gold layer (2 checks)

| Check | Rule | Status thresholds |
|---|---|---|
| **Call summary grain** | `gold_vessel_call_summary` must have exactly one row per `vessel_call_id`. Duplicates indicate a fan-out join bug in gold production. | > 0 dupes → FAIL |
| **KPI grain** | `gold_daily_terminal_kpis` must have exactly one row per `(terminal_id, operation_date)`. Duplicates would double-count KPIs in dashboards. | > 0 dupes → FAIL |

### ML feature table (2 checks)

| Check | Rule | Status thresholds |
|---|---|---|
| **Feature null rates** | Static and temporal features (terminal, temporal, weather, planned scope) must have 0% nulls. Rolling history features (`avg_previous_10_*`, `previous_vessel_delay`) tolerate higher null rates for calls with no qualifying prior history. | Static: > 5% → FAIL. History: > 15% → WARN, > 50% → FAIL |
| **Target distribution** | Positive label rate below 5% means the model has almost nothing to learn; above 60% means the class imbalance requires special handling beyond `class_weight`. | < 5% or > 60% → WARN, > 95% → FAIL |

### Cross-layer (1 check)

| Check | Rule | Status thresholds |
|---|---|---|
| **Row count reconciliation** | Gold summary distinct call IDs must equal silver. ML features must be a subset of silver. ML features having more IDs than silver would indicate fabricated rows — a data integrity failure. | Gold ≠ Silver → WARN. ML > Silver → FAIL |

---

## Outlier Detection: Tukey IQR Method

The `large_delay_outlier_flag` uses the **Tukey upper fence**:

```
fence = Q3 + 1.5 × (Q3 − Q1)
```

computed on **positive delays only** (zeros are excluded — on-time calls would artificially inflate Q1 and compress the IQR). The fence is floored at **24 hours** so that isolated short delays never set a trivially low threshold that would flag most of the dataset.

This is the same method implemented in both `src/silver_transformations.py` and `sql/03_data_quality_checks.sql` (Check 9). Running both implementations and comparing their outputs is a cross-validation of correctness.

---

## Expected Dirty Data

The synthetic generator deliberately injects quality issues at approximately 4% total rate:

| Issue | Count | Rate | Detected by |
|---|---|---|---|
| Missing ETA | 30 | 1.5% | `missing_eta_flag` |
| Missing ATA | 20 | 1.0% | `missing_ata_flag` |
| Invalid timestamp sequences | 10 | 0.5% | `invalid_arrival_sequence_flag` |
| Duplicate call records | 20 | 1.0% | Deduplication (removed, not flagged) |
| Late correction updates | 8 | 0.4% | Deduplication (most-recent kept) |
| Crane time inversions | 148 | 2.0% of assignments | `invalid_crane_time_flag` |
| Crane schedule overlaps | 2,796 | 37.7% of assignments | `crane_overlap_flag` |
| Negative move durations | 284 | 0.3% of moves | Nulled in silver |

The high crane overlap rate (37.7%) is by design — it represents a real operational scenario where crane scheduling data has not been reconciled with actual allocation changes. In a production system, this rate would trigger an immediate alert and investigation. The DQ check framework correctly classifies it as FAIL.

---

## DQ Report Output

`python src/data_quality_checks.py` writes `outputs/data_quality_report.json` with a structured payload:

```json
{
  "run_timestamp": "...",
  "overall_status": "FAIL | WARN | PASS",
  "summary": { "total_checks": 14, "passed": 12, "warned": 1, "failed": 1 },
  "checks": {
    "<check_name>": {
      "status": "FAIL",
      "description": "...",
      "value": { ... },
      "threshold": { ... },
      "detail": { ... }
    }
  }
}
```

This format is designed to be consumed by a downstream alerting system (e.g., a Slack webhook or PagerDuty integration that fires on `overall_status == "FAIL"`).
