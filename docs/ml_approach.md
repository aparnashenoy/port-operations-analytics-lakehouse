# ML Approach

## Problem Statement

Given an ETA notification for an upcoming vessel call, predict whether the vessel will arrive more than two hours late. The prediction must be made using only information available at the time of the notification — before the vessel arrives.

This is a binary classification problem. The label `is_arrival_delayed_more_than_2_hours` is 1 when `arrival_delay_minutes > 120`, else 0. It is null for in-progress calls with no recorded ATA; those rows are inference targets, not training rows.

---

## Point-in-Time Correctness

Point-in-time (PIT) correctness is the central constraint of this feature set. It means every feature value must represent a quantity that was knowable at the moment the ETA notification arrived — not at the time the data was extracted from the database.

**Why this matters:** if a model is trained with features computed from the full dataset (including data that postdates the call being predicted), it learns patterns that do not exist at inference time. The trained model produces optimistic metrics in evaluation but systematically fails in production. This failure mode is called **target leakage** or **temporal data leakage**.

The canonical example: using `actual_turnaround_hours` (which includes the actual port stay) as a feature for predicting whether the vessel will be delayed. This column is only knowable after the vessel has already arrived and departed — exactly the event being predicted.

**How PIT is enforced in this pipeline:**

For rolling history features, a cross-merge pattern is used:

```python
merged = candidates.merge(history, on="group_col", how="left")
merged = merged[
    (merged["_h_atd"] < merged["eta"]) &        # strict: history must predate current ETA
    (merged["_h_call_id"] != merged["vessel_call_id"])  # exclude self
]
```

The strict `<` inequality (not `<=`) ensures no call can use information from another call that completed at the exact same timestamp.

For congestion features, only `eta` and `etd` (planned times) from other calls are used — never `ata` or `atd` (actuals):

```python
congestion_mask = (
    (other_eta <= current_eta) &   # other call planned to be in berth
    (other_etd >= current_eta)     # other call planned to still be there
)
```

---

## Feature Groups

### A. Categorical identifiers (encoded)

| Feature | PIT justification |
|---|---|
| `terminal_id_encoded` | Terminal is fixed at booking time |
| `service_code_encoded` | Carrier/service is fixed at booking time |

Encoded using `OrdinalEncoder` with fixed category lists (`TERMINAL_CATEGORIES`, `SERVICE_CATEGORIES`). The lists are hardcoded in `feature_engineering.py` to guarantee identical integer mappings across separate training and inference runs. Unknown categories encode to -1 (handled by `handle_unknown="use_encoded_value"`).

### B. Static call scope (known at booking)

| Feature | PIT justification |
|---|---|
| `vessel_capacity_teu` | Planned cargo volume is confirmed at booking |
| `planned_moves` | Container move plan is set before arrival |
| `planned_crane_count` | Crane bookings are made before arrival |
| `planned_crane_hours` | Derived from `planned_start`/`planned_end` in crane assignments |

`planned_crane_hours` is derived from the **planned** timestamps (`planned_start`, `planned_end`), not from `actual_start`/`actual_end`, which are only available after the operation concludes.

### C. Temporal (derived from ETA)

| Feature | PIT justification |
|---|---|
| `day_of_week` | ETA is known at notification time |
| `month` | Same |
| `is_weekend` | Same |

Weekend arrivals face different pilot and berth crew availability — a known operational pattern.

### D. Weather (observable at ETA date)

| Feature | PIT justification |
|---|---|
| `storm_flag` | Weather forecast for the ETA calendar day is observable before arrival |

`storm_flag = 1` when `severity_index > 0.5`. Weather severity is joined on `(terminal_id, date(eta))`, not on the actual arrival date.

### E. Historical rolling features (PIT self-join)

| Feature | PIT justification |
|---|---|
| `avg_previous_10_terminal_delays` | Mean delay of the 10 most recent completed calls at this terminal with `atd < current_eta` |
| `avg_previous_10_service_delays` | Same, grouped by carrier service |
| `previous_vessel_delay` | Single most recent delay for this vessel with `atd < current_eta` |

These three features have the most predictive power (collectively ~38% of RF importance) because they capture current congestion state and carrier-specific operational patterns. Calls early in the simulation window receive NaN for these features — there is no qualifying history — and are imputed at training time with the column median.

### F. Congestion (planned schedule)

| Feature | PIT justification |
|---|---|
| `terminal_congestion_score` | Count of other calls whose planned window (`eta`...`etd`) overlaps the current call's ETA. Uses planned times only. |

---

## Leakage Columns (explicitly excluded)

The following columns are **never** included in the feature set. Each is a post-arrival actual:

- `ata`, `atd` — actual timestamps, known only after vessel arrives/departs
- `arrival_delay_minutes` — derived from ATA; this is what we are predicting
- `departure_delay_minutes` — derived from ATD
- `delay_hours` — same as above
- `actual_turnaround_hours` — derived from ATD and ATA
- `actual_cargo_teu` — recorded after cargo operations complete

These exclusions are enforced at two levels: `assemble_feature_table()` in `feature_engineering.py` selects only named columns (an allowlist, not a denylist), and `test_features.py` has parametrised tests that assert each leakage column is absent from both the feature table and `MODEL_FEATURES`.

---

## Train / Test Split

The dataset is split **chronologically**, not randomly. The last 20% of eligible calls by ETA (post 2024-08-19) form the test set.

**Why not random:** rolling history features are computed from the same call pool used for training. A random split would scatter future calls into training, making their outcomes visible in the history features of training rows — inflating metrics by precisely the mechanism the split is meant to evaluate.

| | Rows | Positive rate |
|---|---|---|
| Train | 1,552 | 26.9% |
| Test | 389 | 25.2% |

The similar positive rates confirm the synthetic data's delay pattern is not concentrated in one time period.

---

## Model Selection

### Logistic Regression (baseline)

**Why:** linear models are the appropriate baseline for any classification task. They are fast, interpretable, and expose whether the features have linear signal at all. A linear model with AUC near 0.5 tells you the features are not sufficient on their own — useful diagnostic information.

**Pipeline:** `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression(C=0.5, class_weight="balanced", solver="liblinear")`.

`StandardScaler` is required because L2 regularisation treats all coefficient magnitudes equally — without scaling, features with large numerical ranges (e.g., `vessel_capacity_teu` in the thousands) would be under-penalised relative to binary flags.

### Random Forest (ensemble)

**Why:** tree-based ensembles capture interaction effects that logistic regression cannot. The combination of `avg_previous_10_terminal_delays` and `storm_flag`, for example, may only matter together — a delayed terminal in a storm is worse than the sum of the two separately. Random forests also handle ordinal-encoded categoricals without requiring one-hot expansion.

**Pipeline:** `SimpleImputer(median)` → `RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced")`.

`min_samples_leaf=10` prevents overfitting on the 1,552-row training set by ensuring every leaf represents at least 10 calls. `n_estimators=300` reduces variance without significant runtime cost.

### Feature Importance

Top 5 features by RF mean decrease in Gini impurity (normalised):

| Feature | RF importance | LR coefficient |
|---|---|---|
| `avg_previous_10_service_delays` | 0.133 | +0.101 |
| `vessel_capacity_teu` | 0.132 | +0.046 |
| `avg_previous_10_terminal_delays` | 0.130 | +0.069 |
| `planned_crane_hours` | 0.126 | +0.076 |
| `planned_moves` | 0.104 | +0.016 |

All five have positive LR coefficients — larger vessels with heavier cargo plans at busier terminals are more likely to be delayed. This is operationally intuitive.

---

## Production Deployment Considerations

1. **Trigger:** in production, the pipeline runs when a new ETA notification arrives from the port management system or via AIS feed. The feature computation runs against the live silver tables.

2. **Inference targets:** rows with `is_arrival_delayed_more_than_2_hours = NaN` (in-progress calls) are the inference population. The trained pipeline (saved as `models/random_forest.joblib`) can score them immediately.

3. **Model refresh:** the rolling history features shift as new completed calls land in silver. The model should be retrained on a rolling window (e.g., trailing 18 months) to keep the history-based features calibrated to current operational conditions.

4. **Threshold tuning:** the default 0.5 decision threshold may not be optimal. In an operations alert context, the cost of a missed delay is higher than the cost of a false alarm — a lower threshold (e.g., 0.3) increases recall at the cost of precision, which may be preferable.

5. **Monitoring:** feature drift (particularly in the rolling history features) should be monitored via a population stability index (PSI) or Kolmogorov–Smirnov test on the marginal distributions of `avg_previous_10_terminal_delays` and `vessel_capacity_teu`.
