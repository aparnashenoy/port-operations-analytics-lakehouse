# Data Model

## Entity Overview

The model describes the operational lifecycle of a vessel call: a cargo vessel arrives at a terminal, berths, has cranes assigned to work its cargo, individual container moves are performed, and the vessel departs. Weather affects all phases. Each terminal has fixed physical characteristics that constrain throughput.

---

## Table Inventory

### Silver layer (source of truth for analysts)

#### `vessel_calls_silver`

**Grain:** one row per vessel call (a single visit by one vessel to one terminal).

| Column | Type | Description |
|---|---|---|
| `vessel_call_id` | string | Primary key. Surrogate ID assigned by the port system |
| `vessel_id` | string | Identifies the physical vessel across multiple calls |
| `vessel_name` | string | Vessel name |
| `vessel_class` | string | Size class: Sub-Panamax, Panamax, Post-Panamax, ULCV |
| `carrier_code` | string | Shipping line operating the service (10 carriers) |
| `terminal_id` | string | Which of the five terminals handled this call |
| `berth_id` | string | Physical berth within the terminal |
| `eta` | timestamp | Estimated time of arrival (scheduled) |
| `ata` | timestamp | Actual time of arrival |
| `etd` | timestamp | Estimated time of departure |
| `atd` | timestamp | Actual time of departure |
| `delay_hours` | float | Arrival delay in hours (ATA − ETA); positive = late |
| `delay_reason` | string | Categorical reason code when delay > 0 |
| `planned_turnaround_hours` | float | Planned port time (ETD − ETA) |
| `actual_turnaround_hours` | float | Actual port time (ATD − ATA) |
| `planned_cargo_teu` | integer | Planned container volume in TEU |
| `actual_cargo_teu` | integer | Actual container volume handled |
| `weather_impact_factor` | float | Multiplier representing weather drag on operations (1.0 = no impact) |
| `status` | string | `completed`, `in_progress`, `scheduled` |
| `arrival_delay_minutes` | float | Derived: (ATA − ETA) in minutes. Null if either timestamp missing |
| `departure_delay_minutes` | float | Derived: (ATD − ETD) in minutes |
| `missing_eta_flag` | boolean | DQ flag: ETA is null |
| `missing_ata_flag` | boolean | DQ flag: ATA is null |
| `invalid_arrival_sequence_flag` | boolean | DQ flag: ATA > ATD (physically impossible) |
| `invalid_departure_sequence_flag` | boolean | DQ flag: ETD < ETA (scheduling error) |
| `large_delay_outlier_flag` | boolean | DQ flag: delay exceeds Tukey upper fence (≥ 24 h floor) |

**Key relationships:**
- → `crane_assignments_silver` via `vessel_call_id` (one-to-many)
- → `container_moves_silver` via `vessel_call_id` (one-to-many)
- → `weather_daily_silver` via `(terminal_id, date(eta))` (many-to-one)
- → `terminal_metadata_silver` via `terminal_id` (many-to-one)

---

#### `crane_assignments_silver`

**Grain:** one row per crane assignment (one crane serving one vessel call for one continuous block of time).

| Column | Type | Description |
|---|---|---|
| `assignment_id` | string | Primary key |
| `vessel_call_id` | string | FK → vessel_calls |
| `crane_id` | string | Physical crane identifier |
| `terminal_id` | string | Terminal (denormalised from vessel_call) |
| `planned_start` / `planned_end` | timestamp | Scheduled crane window |
| `actual_start` / `actual_end` | timestamp | Recorded crane window |
| `planned_hours` | float | Derived: (planned_end − planned_start) / 3600 |
| `actual_crane_hours` | float | Derived: (actual_end − actual_start) / 3600 |
| `productivity_moves_per_hour` | float | Gross crane productivity during this assignment |
| `invalid_crane_time_flag` | boolean | DQ flag: actual_end < actual_start |
| `crane_overlap_flag` | boolean | DQ flag: overlaps another assignment on the same crane |

A single vessel call typically has 2–4 crane assignments in parallel. A crane cannot physically serve two calls simultaneously — overlapping assignments are data quality issues.

---

#### `container_moves_silver`

**Grain:** one row per individual container move (one container loaded or discharged in one operation).

| Column | Type | Description |
|---|---|---|
| `move_id` | string | Primary key |
| `assignment_id` | string | FK → crane_assignments |
| `vessel_call_id` | string | FK → vessel_calls (denormalised) |
| `crane_id` | string | Which crane performed this move |
| `move_type` | string | `load` or `discharge` |
| `container_size` | string | `20ft` or `40ft` |
| `container_type` | string | `dry`, `reefer`, `hazmat`, `open_top` |
| `planned_duration_minutes` | float | Scheduled cycle time for this move |
| `actual_duration_minutes` | float | Recorded cycle time (null if negative — cleaned in silver) |
| `move_status` | string | `completed`, `cancelled` |
| `duration_variance_minutes` | float | actual − planned; positive = slower than planned |

At 94,678 moves across 2,000 calls, this is the highest-volume table (~47 moves per call). Cancelled moves retain a row but have null `actual_move_time`.

---

#### `weather_daily_silver`

**Grain:** one row per (terminal, calendar day). Covers 2023-01-01 to 2024-12-31 across all five terminals.

| Column | Type | Description |
|---|---|---|
| `weather_date` | date | Calendar day (normalised to midnight) |
| `terminal_id` | string | Terminal location |
| `wind_speed_knots` | float | Daily mean wind speed |
| `wave_height_m` | float | Significant wave height in metres |
| `visibility_nm` | float | Visibility in nautical miles |
| `precipitation_mm` | float | Daily precipitation |
| `severity_index` | float | Composite score 0.0–1.0 (higher = worse conditions) |
| `weather_impact_factor` | float | Operational multiplier applied to turnaround estimates |

Severity follows a seasonal pattern: Northern European winters (Nov–Feb) are 1.5× more severe on average than summer months, modelled via a cosine curve.

---

#### `terminal_metadata_silver`

**Grain:** one row per terminal (static reference table, 5 rows).

| Terminal | Name | Berths | Cranes | Max class |
|---|---|---|---|---|
| CPT | Central Port Terminal | 4 | 7 | Panamax |
| EFT | Eastport Freight Terminal | 4 | 8 | Post-Panamax |
| NCT | Northgate Container Terminal | 6 | 12 | ULCV |
| SLH | Southside Logistics Hub | 3 | 5 | Panamax |
| WIT | Westquay Industrial Terminal | 3 | 4 | Sub-Panamax |

NCT is the primary deep-water terminal capable of handling Ultra Large Container Vessels (>18,000 TEU).

---

### Gold layer (aggregated, consumption-ready)

#### `gold_vessel_call_summary`

**Grain:** one row per vessel call. Joins silver KPIs, crane aggregates, and move counts into a single flat table for reporting.

Key columns: `vessel_call_id`, `eta/ata/etd/atd`, `arrival_delay_minutes`, `departure_delay_minutes`, `planned_moves`, `actual_moves`, `total_crane_hours`, `moves_per_crane_hour`, `storm_flag`, `delay_category` (`early` / `on_time` / `minor_delay` / `moderate_delay` / `major_delay`).

#### `gold_daily_terminal_kpis`

**Grain:** one row per (terminal, operation_date). Powers time-series dashboards and trend reporting.

Key columns: `operation_date`, `terminal_id`, `vessel_calls`, `avg_arrival_delay_minutes`, `delayed_vessel_percentage`, `total_container_moves`, `avg_moves_per_crane_hour`, `storm_days_count`, `data_quality_issue_count`.

#### `ml_vessel_delay_features`

**Grain:** one row per eligible vessel call (ETA present and no invalid arrival sequence; 1,960 of 2,000 calls). This is the training and inference dataset — distinct from the operational summary because every column is point-in-time correct. See `docs/ml_approach.md` for the leakage constraints that determine which columns appear here.

---

## Key Business Rules Encoded in the Model

1. **A vessel call is "delayed" when `delay_hours > 0.5`** (30 minutes) for operational tracking, or **`arrival_delay_minutes > 120`** (2 hours) for the ML binary label. These are two distinct thresholds with different use cases.
2. **Crane overlaps are flagged, not resolved.** Two assignments on the same crane in overlapping windows may reflect a scheduling error or a data transmission issue; which is which requires human review.
3. **Gold KPI table uses `operation_date = coalesce(date(ata), date(eta))`** so in-progress calls (no ATA) still appear in daily reporting.
4. **Duplicate vessel calls in bronze are deduplicated using latest-record-wins semantics** (sort by `record_created_at` DESC, `_ingestion_ts` DESC). This handles the common case of source systems re-transmitting corrected records.
