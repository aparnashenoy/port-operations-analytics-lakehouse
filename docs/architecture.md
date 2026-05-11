# Architecture

## Business Context

Container terminals are capital-intensive facilities where predictability is everything. A single delayed vessel call ripples forward: cranes miss their windows, trucks queue at gates, stack plans collapse, and connecting vessels on downstream legs accumulate their own delays. The cost is measurable — berth hire rates for large container ships run $50,000–$100,000 per day, and terminal productivity is benchmarked globally.

This project simulates the analytics stack an operations team would build to answer three questions:

1. **Retrospective** — Where are our delays coming from, and which terminals or carriers are systematically worse?
2. **Real-time monitoring** — Which calls are clean and which are flagged for investigation right now?
3. **Pre-arrival prediction** — Given a vessel's ETA notification, how likely is it to arrive more than two hours late?

The dataset covers five terminals across a 24-month window, tracking 2,000 vessel calls, 7,415 crane assignments, and 94,678 container moves from ten major carriers (APL, CMA CGM, COSCO, Evergreen, Hapag-Lloyd, Maersk, MSC, ONE, PIL, ZIM).

---

## Medallion Architecture

The project follows the **Bronze → Silver → Gold** medallion pattern, originally developed for Apache Spark lakehouse deployments but equally appropriate at smaller scale with Pandas and DuckDB. Each layer has a defined contract; downstream consumers only ever read from the next-higher layer.

```
data/raw/           ← CSV exports from source systems
data/bronze/        ← Raw ingestion with metadata stamps, zero transformation
data/silver/        ← Cleaned, deduplicated, flagged, enriched
data/gold/          ← Aggregated KPIs and ML-ready feature tables
data/quarantine/    ← Rows rejected at bronze (null primary keys)
```

### Bronze layer

**Contract:** every source record lands here, unchanged except for ingestion metadata (`_ingestion_ts`, `_batch_id`, `_source_file`, `_source_system`). Bronze is append-only. Dirty records (duplicate vessel call IDs, missing ETAs, invalid timestamps) are preserved — removing them at this layer would destroy the audit trail.

**Why:** source systems are imperfect. Bronze is the forensic record of what actually arrived. If silver ever produces a wrong answer, bronze lets you replay from first principles.

### Silver layer

**Contract:** one row per business entity, quality-flagged, timestamp-derived metrics computed, no information dropped. Deduplication uses latest-record-wins semantics keyed on `record_created_at` DESC. Quality flags (`missing_eta_flag`, `invalid_arrival_sequence_flag`, `crane_overlap_flag`, etc.) mark problems without deleting the row — analysts see the full distribution, not an artificially clean subset.

**Why:** flag-not-drop lets operations teams investigate data quality issues rather than discovering them silently in production reports.

### Gold layer

**Contract:** denormalised, aggregated, business-consumable. Two tables: a vessel-call summary (one row per call, with KPIs joined in) and a daily terminal KPI series (one row per terminal per day). A third gold table is the ML feature set — produced by `feature_engineering.py` and distinct from operational KPIs because it enforces point-in-time correctness.

**Why:** analysts and dashboard tools should never need to write joins. Gold tables are optimised for reading, not writing.

---

## Technology Stack

| Component | Tool | Rationale |
|---|---|---|
| Storage | Apache Parquet (Snappy) | Columnar, splittable, language-agnostic; reads 10× faster than CSV for analytical queries |
| Transformation | Pandas 2.x | Idiomatic Python; sufficient for this scale; straightforward to test |
| SQL showcase | DuckDB | In-process analytical SQL with full window function support; reads Parquet natively; zero infrastructure |
| ML | scikit-learn Pipelines | Industry-standard; pipeline API enforces consistent train/inference pre-processing |
| Testing | pytest | Standard; parametrised tests for schema contracts |
| Serialisation | PyArrow | Schema enforcement; metadata preservation across Parquet reads |

**Scale note:** this stack runs on a laptop in under 60 seconds for 2,000 vessel calls. The same medallion logic scales horizontally to Spark/Delta Lake or Databricks without changing the business rules — only the execution engine changes.

---

## Pipeline Execution Order

```
python src/generate_data.py          # 1. Emit synthetic CSVs
python src/bronze_ingestion.py       # 2. CSV → Bronze Parquet
python src/silver_transformations.py # 3. Bronze → Silver (clean, flag)
python src/gold_kpis.py              # 4. Silver → Gold KPIs
python src/feature_engineering.py   # 5. Silver + Gold → ML features
python src/train_model.py            # 6. Train and evaluate models
python src/data_quality_checks.py   # 7. Run DQ suite, emit report
pytest tests/                        # 8. Unit + integration tests
```

Each script is independently runnable and reads only from the layer below it. There are no circular dependencies.

---

## SQL Showcase Files

Five standalone DuckDB SQL files in `sql/` implement the same logic as the Python pipeline, expressed as pure SQL for independent auditability. They demonstrate senior-level patterns: `QUALIFY` for deduplication, `GROUPING SETS` for rollup reporting, `LATERAL` subqueries for correlated aggregation, `RANGE BETWEEN INTERVAL` for time-windowed rolling averages, and point-in-time self-joins for ML features. See `sql/README` comments at the top of each file.
