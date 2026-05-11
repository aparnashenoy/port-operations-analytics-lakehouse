-- =============================================================================
-- 03_data_quality_checks.sql
-- Data quality validation suite for the port operations lakehouse.
--
-- Each check targets a specific failure category and returns zero rows when
-- the data is clean.  Non-zero results indicate records that require
-- investigation before promotion to the next medallion layer.
--
-- Checks
--   1   Duplicate vessel call detection
--   2   Null rate audit across all key columns
--   3   Referential integrity: orphaned foreign keys
--   4   Invalid timestamp sequences
--   5   Turnaround time out-of-bounds
--   6   Cargo volume anomalies
--   7   Crane assignment overlap conflicts
--   8   Container move status consistency
--   9   Statistical outlier detection (IQR method)
--  10   Cross-layer record count reconciliation
--  11   DQ scorecard — one summary row per table
-- =============================================================================

CREATE OR REPLACE VIEW bz_vessel_calls AS
    SELECT * FROM read_parquet('data/bronze/vessel_calls.parquet');
CREATE OR REPLACE VIEW sv_vessel_calls AS
    SELECT * FROM read_parquet('data/silver/vessel_calls_silver.parquet');
CREATE OR REPLACE VIEW sv_crane_assignments AS
    SELECT * FROM read_parquet('data/silver/crane_assignments_silver.parquet');
CREATE OR REPLACE VIEW sv_container_moves AS
    SELECT * FROM read_parquet('data/silver/container_moves_silver.parquet');


-- ─── Check 1: Duplicate vessel calls ─────────────────────────────────────────
-- Expectation: zero rows after silver deduplication.
-- A non-zero result means the silver dedup step did not run or failed.

SELECT
    vessel_call_id,
    COUNT(*)           AS occurrences,
    MIN(ata)           AS first_ata,
    MAX(record_created_at) AS latest_update
FROM sv_vessel_calls
GROUP BY vessel_call_id
HAVING COUNT(*) > 1
ORDER BY occurrences DESC;

-- Bronze duplicate summary (expected: some duplicates exist as injected issues)
SELECT
    COUNT(*)                                        AS total_bronze_rows,
    COUNT(DISTINCT vessel_call_id)                  AS unique_call_ids,
    COUNT(*) - COUNT(DISTINCT vessel_call_id)       AS duplicate_rows,
    ROUND(100.0 * (COUNT(*) - COUNT(DISTINCT vessel_call_id))
          / COUNT(*), 2)                            AS duplicate_rate_pct
FROM bz_vessel_calls;


-- ─── Check 2: Null rate audit ─────────────────────────────────────────────────
-- Reports the null count and null rate for every column that carries a
-- non-trivial null rate.  Mandatory columns with nulls are highlighted.

WITH null_counts AS (
    SELECT
        'vessel_call_id'          AS column_name, TRUE  AS mandatory,
            COUNT(*) FILTER (WHERE vessel_call_id IS NULL)          AS null_count,
            COUNT(*)                                                 AS total
    FROM sv_vessel_calls

    UNION ALL SELECT 'vessel_id',  TRUE,
            COUNT(*) FILTER (WHERE vessel_id IS NULL), COUNT(*) FROM sv_vessel_calls
    UNION ALL SELECT 'terminal_id', TRUE,
            COUNT(*) FILTER (WHERE terminal_id IS NULL), COUNT(*) FROM sv_vessel_calls
    UNION ALL SELECT 'eta',        FALSE,
            COUNT(*) FILTER (WHERE eta IS NULL), COUNT(*) FROM sv_vessel_calls
    UNION ALL SELECT 'ata',        FALSE,
            COUNT(*) FILTER (WHERE ata IS NULL), COUNT(*) FROM sv_vessel_calls
    UNION ALL SELECT 'etd',        FALSE,
            COUNT(*) FILTER (WHERE etd IS NULL), COUNT(*) FROM sv_vessel_calls
    UNION ALL SELECT 'atd',        FALSE,
            COUNT(*) FILTER (WHERE atd IS NULL), COUNT(*) FROM sv_vessel_calls
    UNION ALL SELECT 'delay_reason', FALSE,
            COUNT(*) FILTER (WHERE delay_reason IS NULL), COUNT(*) FROM sv_vessel_calls
    UNION ALL SELECT 'planned_cargo_teu', FALSE,
            COUNT(*) FILTER (WHERE planned_cargo_teu IS NULL), COUNT(*) FROM sv_vessel_calls
)
SELECT
    column_name,
    mandatory,
    null_count,
    total,
    ROUND(100.0 * null_count / NULLIF(total, 0), 2)           AS null_rate_pct,
    CASE
        WHEN mandatory AND null_count > 0 THEN 'FAIL — mandatory column has nulls'
        WHEN null_count = 0               THEN 'PASS'
        ELSE                                   'WARN — optional column has nulls'
    END                                                        AS check_result
FROM null_counts
WHERE null_count > 0
ORDER BY mandatory DESC, null_rate_pct DESC;


-- ─── Check 3: Referential integrity ──────────────────────────────────────────
-- Crane assignments must reference a vessel_call_id that exists in silver
-- vessel_calls.  Orphaned assignments indicate a join key mismatch or a
-- vessel call that was quarantined without cascading the delete.

SELECT
    ca.assignment_id,
    ca.vessel_call_id,
    ca.terminal_id,
    ca.actual_start
FROM sv_crane_assignments ca
LEFT JOIN sv_vessel_calls vc USING (vessel_call_id)
WHERE vc.vessel_call_id IS NULL   -- assignment has no matching vessel call
ORDER BY ca.assignment_id;

-- Count summary
SELECT
    COUNT(*)                                        AS total_crane_assignments,
    COUNT(*) FILTER (WHERE vc.vessel_call_id IS NULL)
                                                    AS orphaned_assignments,
    ROUND(100.0 *
          COUNT(*) FILTER (WHERE vc.vessel_call_id IS NULL)
          / COUNT(*), 2)                            AS orphan_rate_pct
FROM sv_crane_assignments ca
LEFT JOIN sv_vessel_calls vc USING (vessel_call_id);


-- ─── Check 4: Invalid timestamp sequences ────────────────────────────────────
-- These rows carry invalid_arrival_sequence_flag = TRUE in silver.
-- This check confirms the flag is correctly set and returns the offending rows.

SELECT
    vessel_call_id,
    terminal_id,
    eta,
    ata,
    atd,
    datediff('minute', atd, ata)    AS ata_after_atd_by_minutes,
    invalid_arrival_sequence_flag
FROM sv_vessel_calls
WHERE invalid_arrival_sequence_flag = TRUE
ORDER BY vessel_call_id;

-- Cross-check: flag count matches raw timestamp condition
SELECT
    COUNT(*) FILTER (WHERE invalid_arrival_sequence_flag)       AS flagged_count,
    COUNT(*) FILTER (WHERE ata IS NOT NULL
                       AND atd IS NOT NULL
                       AND ata > atd)                           AS raw_condition_count,
    COUNT(*) FILTER (WHERE invalid_arrival_sequence_flag)
        = COUNT(*) FILTER (WHERE ata IS NOT NULL
                             AND atd IS NOT NULL
                             AND ata > atd)                     AS flag_matches_condition
FROM sv_vessel_calls;


-- ─── Check 5: Turnaround time out-of-bounds ──────────────────────────────────
-- Calls with actual_turnaround_hours outside [1, 168] are operationally
-- implausible.  The silver table preserves them; this check surfaces them
-- for the operations team to review.

SELECT
    vessel_call_id,
    terminal_id,
    vessel_class,
    ata,
    atd,
    ROUND(actual_turnaround_hours, 1)       AS actual_turnaround_hours,
    CASE
        WHEN actual_turnaround_hours < 1    THEN 'too_short'
        WHEN actual_turnaround_hours > 168  THEN 'too_long'
    END                                     AS violation_type
FROM sv_vessel_calls
WHERE actual_turnaround_hours < 1
   OR actual_turnaround_hours > 168
ORDER BY actual_turnaround_hours DESC;


-- ─── Check 6: Cargo volume anomalies ─────────────────────────────────────────
-- Actual cargo should not exceed the vessel's declared capacity.
-- We proxy capacity via planned_cargo_teu (the fill plan is already
-- capacity-constrained by the generator).  Any actual > 1.1× planned
-- is a likely data entry error.

SELECT
    vessel_call_id,
    vessel_class,
    planned_cargo_teu,
    actual_cargo_teu,
    ROUND(100.0 * actual_cargo_teu / NULLIF(planned_cargo_teu, 0), 1)
                                            AS actual_vs_planned_pct
FROM sv_vessel_calls
WHERE actual_cargo_teu > planned_cargo_teu * 1.10
  AND planned_cargo_teu IS NOT NULL
  AND actual_cargo_teu  IS NOT NULL
ORDER BY actual_vs_planned_pct DESC;


-- ─── Check 7: Crane overlap conflicts ────────────────────────────────────────
-- Rows where the crane_overlap_flag is TRUE are listed with their overlap
-- partners so operations can identify which vessel calls are affected.
-- The overlap flag was computed in silver using a LAG + running MAX approach.

WITH overlapping AS (
    SELECT
        ca.crane_id,
        ca.assignment_id,
        ca.vessel_call_id,
        ca.actual_start,
        ca.actual_end,
        -- Identify which prior assignment this one overlaps
        LAG(ca.assignment_id) OVER (
            PARTITION BY ca.crane_id
            ORDER BY ca.actual_start
        ) AS prior_assignment_id,
        LAG(ca.actual_end) OVER (
            PARTITION BY ca.crane_id
            ORDER BY ca.actual_start
        ) AS prior_actual_end
    FROM sv_crane_assignments ca
    WHERE ca.crane_overlap_flag = TRUE
      AND ca.actual_start IS NOT NULL
)
SELECT
    crane_id,
    assignment_id,
    vessel_call_id,
    actual_start,
    actual_end,
    prior_assignment_id,
    prior_actual_end,
    -- Gap is negative when there is an overlap
    datediff('minute', actual_start, prior_actual_end)  AS overlap_minutes
FROM overlapping
WHERE prior_actual_end IS NOT NULL
ORDER BY crane_id, actual_start
LIMIT 20;


-- ─── Check 8: Container move status consistency ───────────────────────────────
-- Every cancelled move should have a NULL actual_move_time.
-- Every completed move should have a non-NULL actual_move_time.

SELECT
    move_status,
    COUNT(*)                                                AS total_moves,
    COUNT(*) FILTER (WHERE actual_move_time IS NULL)        AS null_actual_time,
    COUNT(*) FILTER (WHERE actual_duration_minutes IS NULL) AS null_actual_duration,
    COUNT(*) FILTER (WHERE actual_duration_minutes < 0)     AS negative_duration,
    ROUND(100.0 *
          COUNT(*) FILTER (WHERE actual_move_time IS NULL)
          / COUNT(*), 1)                                    AS null_time_rate_pct
FROM sv_container_moves
GROUP BY move_status
ORDER BY move_status;


-- ─── Check 9: Statistical outlier detection ───────────────────────────────────
-- Identify delay_hours values that exceed the Tukey upper fence (Q3 + 1.5×IQR)
-- computed on positive delays only.  These align with large_delay_outlier_flag
-- in silver but are recalculated here to verify the Python implementation.

WITH stats AS (
    SELECT
        QUANTILE_CONT(delay_hours, 0.25) AS q1,
        QUANTILE_CONT(delay_hours, 0.75) AS q3
    FROM sv_vessel_calls
    WHERE delay_hours > 0
),
fence AS (
    SELECT
        q1,
        q3,
        q3 - q1                       AS iqr,
        GREATEST(q3 + 1.5 * (q3 - q1), 24.0)  AS upper_fence
    FROM stats
)
SELECT
    vc.vessel_call_id,
    vc.terminal_id,
    vc.delay_reason,
    ROUND(vc.delay_hours, 2)          AS delay_hours,
    ROUND(f.upper_fence, 2)           AS outlier_threshold_hours,
    vc.large_delay_outlier_flag       AS flagged_in_silver
FROM sv_vessel_calls vc
CROSS JOIN fence f
WHERE vc.delay_hours > f.upper_fence
ORDER BY vc.delay_hours DESC;


-- ─── Check 10: Cross-layer record count reconciliation ────────────────────────
-- The number of distinct vessel_call_ids must be consistent across layers.
-- Bronze has more rows than silver (duplicates removed); silver and gold
-- should have the same call count.

SELECT
    layer,
    row_count,
    distinct_call_ids
FROM (
    SELECT 'bronze' AS layer,
           COUNT(*)                         AS row_count,
           COUNT(DISTINCT vessel_call_id)   AS distinct_call_ids
    FROM bz_vessel_calls

    UNION ALL BY NAME

    SELECT 'silver' AS layer,
           COUNT(*)                         AS row_count,
           COUNT(DISTINCT vessel_call_id)   AS distinct_call_ids
    FROM sv_vessel_calls

    UNION ALL BY NAME

    SELECT 'gold' AS layer,
           COUNT(*)                         AS row_count,
           COUNT(DISTINCT vessel_call_id)   AS distinct_call_ids
    FROM read_parquet('data/gold/gold_vessel_call_summary.parquet')
)
ORDER BY layer;


-- ─── Check 11: DQ scorecard ───────────────────────────────────────────────────
-- One summary row per silver table.  Use this as a pipeline health dashboard:
-- if any score drops below the configured threshold the pipeline should alert.

SELECT
    'vessel_calls' AS table_name,
    COUNT(*)       AS total_rows,

    ROUND(100.0 * COUNT(*) FILTER (WHERE missing_eta_flag)                / COUNT(*), 2) AS missing_eta_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE missing_ata_flag)                / COUNT(*), 2) AS missing_ata_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE invalid_arrival_sequence_flag)   / COUNT(*), 2) AS invalid_seq_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE large_delay_outlier_flag)        / COUNT(*), 2) AS outlier_pct,

    -- Overall DQ score: % of rows with zero flags
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE NOT missing_eta_flag
          AND NOT missing_ata_flag
          AND NOT invalid_arrival_sequence_flag
          AND NOT large_delay_outlier_flag
    ) / COUNT(*), 1)                                                       AS clean_row_pct

FROM sv_vessel_calls

UNION ALL BY NAME

SELECT
    'crane_assignments' AS table_name,
    COUNT(*)            AS total_rows,
    NULL AS missing_eta_pct,
    NULL AS missing_ata_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE invalid_crane_time_flag) / COUNT(*), 2) AS invalid_seq_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE crane_overlap_flag)      / COUNT(*), 2) AS outlier_pct,
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE NOT invalid_crane_time_flag
          AND NOT crane_overlap_flag
    ) / COUNT(*), 1)                                                        AS clean_row_pct
FROM sv_crane_assignments;
