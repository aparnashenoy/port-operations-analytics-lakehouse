# Port Operations Analytics Lakehouse

A production-style analytics lakehouse simulating vessel scheduling, berth utilization, and delay prediction for a mid-sized container port. Built to demonstrate Bronze/Silver/Gold medallion architecture, data quality enforcement, analytical SQL patterns, and ML-based delay prediction — all on local tooling with no external dependencies.

---

## Business Context

Port operations teams need reliable intelligence on vessel turnaround times, berth occupancy rates, cargo throughput, and delay root causes to optimize scheduling and minimize demurrage costs. This project simulates that analytical environment using synthetic but operationally realistic data across a 24-month window covering ~50,000 vessel movements.

---

## Architecture

```
data/
├── raw/                  # Synthetic source data (CSV, JSON)
├── bronze/               # Ingested, schema-validated Parquet
├── silver/               # Cleaned, deduplicated, business-keyed Parquet
└── gold/                 # Aggregated analytical tables (DuckDB views)

src/
├── config.py             # All environment and path configuration
├── ingestion/            # Bronze-layer loaders and schema validators
├── transforms/           # Silver-layer cleaning and enrichment
├── analytics/            # Gold-layer DuckDB SQL models
├── ml/                   # Feature engineering and delay prediction
└── quality/              # Great Expectations-style data quality checks

tests/
├── unit/
└── integration/
```

### Medallion Layers

| Layer | Purpose | Format | Tool |
|-------|---------|--------|------|
| Bronze | Raw ingestion with schema enforcement | Parquet | Pandas + PyArrow |
| Silver | Cleaned, deduplicated, joined | Parquet | Pandas |
| Gold | Aggregated KPIs and analytical views | DuckDB in-process | DuckDB SQL |

---

## Key Capabilities

### Data Engineering
- Synthetic data generation covering vessels, berths, port calls, cargo manifests, and weather events
- Schema validation at ingestion with reject/quarantine pattern
- Incremental load simulation with idempotent upsert logic
- Parquet partitioning by port call date for efficient predicate pushdown

### Analytics (DuckDB SQL)
- Berth utilization rate by terminal and vessel class
- Average and P95 vessel turnaround time by carrier and route
- Demurrage exposure by delay root cause (weather, equipment, documentation)
- Weekly cargo throughput trends (TEU volume)
- Delay rate rolling 30-day by berth

### Data Quality
- Null rate checks on mandatory fields
- Referential integrity between vessel registry and port calls
- Turnaround time range validation (reject negative or implausibly long durations)
- Duplicate port call detection with configurable dedup window

### Machine Learning
- Feature engineering: historical delay rate per vessel, seasonal patterns, berth congestion index, weather severity score
- Binary classification: `is_delayed` (turnaround exceeds agreed window by >2 hours)
- Model: Gradient Boosted Trees (scikit-learn) with cross-validated hyperparameter tuning
- Evaluation: PR-AUC, F1 at operating threshold, feature importance report

---

## Quickstart

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data
python -m src.ingestion.generate_data

# 4. Run the full pipeline
python -m src.pipeline run

# 5. Open analytics
python -m src.analytics.run_queries

# 6. Train and evaluate delay model
python -m src.ml.train

# 7. Run tests
pytest tests/ -v
```

---

## Data Model

### Core Entities

**vessels** — registry of vessels that call at the port
| Column | Type | Notes |
|--------|------|-------|
| vessel_id | VARCHAR | IMO-format identifier |
| vessel_name | VARCHAR | |
| vessel_class | VARCHAR | Feeder / Sub-Panamax / Panamax / Post-Panamax |
| carrier_code | VARCHAR | Shipping line |
| flag_state | VARCHAR | ISO 3166-1 alpha-2 |
| capacity_teu | INTEGER | |

**berths** — physical berth registry
| Column | Type | Notes |
|--------|------|-------|
| berth_id | VARCHAR | |
| terminal | VARCHAR | |
| max_vessel_class | VARCHAR | Capacity constraint |
| crane_count | INTEGER | |

**port_calls** — central fact table; one row per vessel visit
| Column | Type | Notes |
|--------|------|-------|
| port_call_id | VARCHAR | Surrogate key |
| vessel_id | VARCHAR | FK → vessels |
| berth_id | VARCHAR | FK → berths |
| eta | TIMESTAMP | Estimated time of arrival |
| ata | TIMESTAMP | Actual time of arrival |
| atd | TIMESTAMP | Actual time of departure |
| delay_reason | VARCHAR | weather / equipment / documentation / none |
| cargo_teu | INTEGER | TEUs handled |

**weather_events** — daily weather severity index per port zone
| Column | Type | Notes |
|--------|------|-------|
| event_date | DATE | |
| zone | VARCHAR | |
| wind_speed_knots | FLOAT | |
| visibility_nm | FLOAT | |
| severity_index | FLOAT | Composite 0–1 |

---

## Design Decisions

- **DuckDB over Spark**: The dataset fits in memory; DuckDB provides OLAP-grade SQL without cluster overhead, making this reproducible on a laptop.
- **Parquet for storage**: Column-oriented format enables efficient predicate pushdown in the gold layer without a metastore.
- **Synthetic data**: Fully parameterized generation ensures no PII risk and allows controlled injection of delay scenarios for ML evaluation.
- **No orchestration framework**: Pipeline is invoked via a simple CLI runner. In production this would be replaced by Airflow or Prefect DAGs.

---

## Project Status

| Component | Status |
|-----------|--------|
| Synthetic data generation | Planned |
| Bronze ingestion pipeline | Planned |
| Silver transforms | Planned |
| Gold DuckDB analytics | Planned |
| Data quality checks | Planned |
| ML — feature engineering | Planned |
| ML — model training | Planned |
| Tests | Planned |

---

## Tech Stack

| Tool | Version | Role |
|------|---------|------|
| Python | 3.11+ | Runtime |
| Pandas | 2.x | DataFrame transforms |
| DuckDB | 0.10+ | In-process OLAP SQL |
| PyArrow | 15+ | Parquet I/O |
| Scikit-learn | 1.4+ | ML pipeline |
| Pytest | 8.x | Testing |
| Faker | 24+ | Synthetic data generation |

---

## License

MIT
