# Interview Talking Points

## 30-Second Explanation

"I built a port operations analytics lakehouse that simulates the data stack a container terminal would use to track and predict vessel delays. It ingests synthetic data for 2,000 vessel calls across five terminals, cleans and flags it through a Bronze/Silver/Gold medallion pipeline, and then trains a classifier to predict whether a vessel will arrive more than two hours late — using only information that would be available before the ship arrives. The whole thing runs end-to-end in under a minute, with 106 automated tests and a 14-check data quality framework."

---

## 2-Minute Explanation

"The project has three layers, each with a defined contract.

Bronze is the raw landing zone — every record from the source systems lands here unchanged, with ingestion metadata stamped on. Nothing is modified or deleted at this layer; it's the audit trail.

Silver is the cleaned, operational truth. Duplicates are resolved using latest-record-wins semantics. Quality problems — missing ETAs, invalid timestamp sequences, crane scheduling conflicts — are marked with boolean flags rather than silently removed. That flag-not-drop approach means analysts see the full distribution of the data, including the dirty records, and can investigate issues rather than discovering them in production.

Gold is the consumption layer: a denormalised vessel call summary, a daily terminal KPI time series, and an ML feature table. The feature table is the interesting one — every column is point-in-time correct, meaning it represents only what an operations analyst would actually know at the moment a vessel sends its ETA notification. Rolling history features use a cross-join filtered to `history.atd < current.eta` with a strict inequality to prevent any future information from leaking into the training set.

The ML side trains a Logistic Regression baseline and a Random Forest, both using sklearn Pipelines to guarantee that imputation and scaling are applied consistently at inference time. The dataset is split chronologically — the last 20% of calls by ETA form the test set — because a random split would scatter future vessel histories into the training rows of earlier calls.

There's also a 14-check DQ framework and 106 pytest tests covering schema contracts, flag logic, PIT leakage prevention, and grain uniqueness across every layer."

---

## Deep-Dive Technical Explanation

### Medallion architecture decisions

The bronze layer is append-only by design. In production, port management systems often re-transmit corrected records — a vessel call with a late ETA update, or a crane assignment with a corrected timestamp. Those corrections arrive as new rows with the same primary key. Bronze preserves both the original and the correction; silver resolves them using `record_created_at DESC` ordering (latest-record-wins). This deduplication pattern is auditable: if silver ever produces a wrong answer, you can replay from bronze to trace the correction history.

The silver-to-gold boundary enforces aggregation contracts. Gold KPI tables are never joined to silver directly in production queries — they are pre-computed so that dashboard tools never need to run complex joins. The fan-out risk (joining crane assignments one-to-many to vessel calls before aggregating) is pre-empted by materialising a `crane_agg_by_call` intermediate before the gold join.

### Point-in-time correctness implementation

The rolling history features use a cross-join pattern:

```python
merged = candidates.merge(history, on="terminal_id", how="left")
merged = merged[
    (merged["_h_atd"] < merged["eta"]) &
    (merged["_h_call_id"] != merged["vessel_call_id"])
]
```

The strict `<` inequality (not `<=`) is intentional. Two calls completing at the exact same timestamp — which can happen in synthetic data and at busy terminals with concurrent operations — must not share each other's outcomes. Using `<=` would allow a call to "see" another call that finished at the same instant, violating PIT.

Congestion features avoid this entirely by using only planned times (`eta`, `etd`) from other calls, never actual arrivals. A count of how many other vessels are scheduled in a berth window at your ETA is knowable before you arrive; a count of how many were actually there is not.

### Data quality framework design

The Tukey IQR method for outlier detection is computed on positive delays only. Including on-time calls (delay = 0) in the quartile computation would artificially inflate Q1 and compress the IQR, producing a fence that flags normal delays as extreme. The fence is also floored at 24 hours so that a cluster of 6-hour delays — which might be locally extreme but are operationally unremarkable — can never trigger a FAIL threshold on a healthy dataset.

The same Tukey computation runs in both `silver_transformations.py` (Python) and `sql/03_data_quality_checks.sql` (DuckDB). Running both and comparing outputs is a cross-validation of correctness — each implementation is an independent check on the other.

### Crane overlap detection

Detecting whether a crane has overlapping assignments is a non-trivial group operation. The approach uses `cummax().shift(1)` within each `crane_id` group sorted by `actual_start`:

```python
df_check["_prior_max_end"] = (
    df_check.groupby("crane_id")["actual_end"]
    .transform(lambda s: s.cummax().shift(1))
)
overlap_mask = df_check["actual_start"] < df_check["_prior_max_end"]
```

`cummax()` tracks the furthest-ending assignment seen so far in the sequence. `shift(1)` lags it by one row so each row compares against the maximum end of all preceding assignments — not including itself. This correctly handles non-contiguous overlaps (where assignment 3 overlaps assignment 1 but not assignment 2).

There is an important edge case: if all assignments in a batch have `actual_end < actual_start` (the invalid_crane_time condition), the valid subset is empty. Running `groupby().transform()` on an empty DataFrame returns a float64 Series instead of datetime64, which causes a TypeError on the timestamp comparison. The production code guards against this with an early-return before the cummax computation.

### Why AUC of 0.538 is the correct result

The synthetic delay variable is generated as a log-normal draw with a weather multiplier and a congestion multiplier. Those multipliers add signal but do not fully determine the outcome — there is substantial residual randomness. An AUC near 0.5 on synthetic data is expected and is not a model failure; it reflects that the features explain the structural drivers of delay (weather, vessel size, terminal congestion) but cannot explain the random component that was deliberately built into the generator. On real port data, where delays have genuine structural causes (pilot availability, tidal windows, berth queue), the same features would produce meaningfully higher AUC.

---

## Possible Interview Questions and Answers

### "Why not use a random train/test split?"

A random split would silently inflate metrics. The rolling history features — `avg_previous_10_terminal_delays`, `previous_vessel_delay` — are computed as look-backs from each call's ETA. If a call in 2024-Q4 is assigned to the training set, its terminal delay history at ETA time includes calls from 2024-Q4 that a random split might put in the test set. The test call's rolling feature then incorporates the outcome of a training row — the test set is no longer held out in any meaningful sense. Chronological splitting ensures the test set contains only calls whose rolling history is computed exclusively from the training period.

### "What is target leakage and how did you prevent it?"

Target leakage is when a feature carries information about the outcome that would not be available at prediction time. In this context, `actual_turnaround_hours` is a leakage column: it is derived from ATD minus ATA, both of which are recorded after the vessel has finished its port stay. Using it would be equivalent to asking the model to predict whether a vessel is delayed using data from after the delay already occurred.

Prevention operates at two levels. First, `assemble_feature_table()` uses an allowlist — it explicitly selects only the named feature columns, so any new column added to silver does not automatically appear in the ML table. Second, the test suite has parametrised tests that assert every known leakage column is absent from both the feature table Parquet file and the `MODEL_FEATURES` list in code. Adding a leakage column would break CI.

### "Why did you choose Random Forest over gradient boosting?"

For a portfolio demonstration with 1,941 labeled rows, a Random Forest is the appropriate choice. Gradient boosted trees (XGBoost, LightGBM) have more hyperparameters and are more sensitive to the learning rate and tree depth settings — they offer higher potential accuracy but require careful cross-validated tuning to realise that potential. `min_samples_leaf=10` in the Random Forest provides a principled regularisation on a small dataset: every leaf represents at least 10 calls, preventing the model from memorising individual vessel quirks. On a production dataset with hundreds of thousands of calls, gradient boosting with Bayesian hyperparameter optimisation would be the right next step.

### "How would you put this model into production?"

The inference path is already structured for production use. The `models/random_forest.joblib` file is a complete sklearn Pipeline that includes the imputer — so calling `pipeline.predict_proba(X)` on new rows handles missing history features identically to training, with no separate preprocessing step required. In production, the trigger would be an incoming AIS message or port system API call with a new ETA notification. The feature computation runs against the live silver tables using the same PIT join logic, and the pipeline scores the row immediately.

The primary operational concern is feature drift. The rolling history features shift as the terminal's delay pattern evolves — a terminal that recovers from a congestion backlog will have lower `avg_previous_10_terminal_delays` values than when the model was trained, and the model's calibration will drift. A population stability index (PSI) check on those features, run weekly, would flag when retraining is warranted.

### "What would you change if this were a real production system?"

Three things. First, distributed execution: the medallion logic ports directly to Spark/Delta Lake — the join patterns, PIT constraints, and DQ flags are all framework-agnostic — but the Pandas implementation caps out around 50 million rows before memory becomes a constraint. Second, streaming ingestion: AIS vessel tracking data arrives as a continuous stream, not a nightly batch; the ETA prediction use case specifically benefits from real-time triggering. Third, threshold tuning: the 0.5 decision threshold is not calibrated to the operations context. In an alert system where a missed delay costs berth re-scheduling and downstream propagation, a threshold of 0.3 that favours recall over precision would be more appropriate — and that threshold should be determined by the operations team's tolerance for false alarms, not by the model's default.

### "Why did you choose DuckDB for the SQL layer?"

DuckDB reads Parquet natively, runs in-process with no server infrastructure, and supports the full analytical SQL feature set — `QUALIFY`, `GROUPING SETS`, `LATERAL`, `RANGE BETWEEN INTERVAL`. It lets the SQL files be fully standalone and runnable without any database setup. In a production context on Databricks or Snowflake, the same SQL patterns translate directly; the SQL showcase is intended to demonstrate query patterns, not infrastructure choice.

---

## Denmark / Nordic Maritime Data Context

The following framing is relevant for roles at Maersk, DSV, DFDS, Port of Aarhus, Greencarrier, or similar logistics and maritime employers.

**Maersk context:** Maersk Technology and Maersk Analytics work on exactly this class of problem — vessel ETA prediction, terminal productivity optimisation, and supply chain visibility. The medallion architecture maps directly to their lakehouse implementations on Azure Databricks. The PIT correctness constraint is particularly relevant: Maersk's route optimisation models have historically suffered from subtle leakage when delay features were computed at report time rather than at notification time.

**Port authority context:** Port of Aarhus and Copenhagen Malmö Port both operate with multi-terminal berth allocation systems. The terminal congestion score feature (`terminal_congestion_score`) directly models the scheduling pressure that berth planners manage manually today. A pre-arrival delay classifier that flags high-risk calls 48 hours in advance enables proactive berth reallocation rather than reactive crisis management.

**DSV / freight forwarding context:** DSV and Scan Global Logistics care about vessel delays because they affect inland transportation bookings and warehouse slot commitments. The `delay_category` field in `gold_vessel_call_summary` (`early`, `on_time`, `minor_delay`, `moderate_delay`, `major_delay`) is the kind of classification their TMS systems consume to trigger automated customer notifications.

**Regulatory and reporting context:** EU Port Services Regulation and the Danish Maritime Authority require ports to report on vessel call performance. The `gold_daily_terminal_kpis` table — with its `delayed_vessel_percentage`, `avg_arrival_delay_minutes`, and `data_quality_issue_count` — is structured to feed directly into those reporting obligations.
