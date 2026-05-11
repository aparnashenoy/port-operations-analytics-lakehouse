-- =============================================================================
-- 01_bronze_to_silver.sql
-- Bronze → Silver transformation logic expressed as DuckDB SQL.
--
-- The Python pipeline (src/silver_transformations.py) is the authoritative
-- implementation.  This file expresses identical logic in pure SQL so the
-- transformation rules are independently auditable, runnable in a DuckDB
-- shell, and reviewable in pull requests without reading Python.
--
-- Sections
--   A  Register bronze Parquet as logical views
--   B  Duplicate detection and classification
--   C  Latest-record deduplication (ROW_NUMBER / QUALIFY)
--   D  Invalid timestamp detection
--   E  Delay metrics and quality flag derivation
--   F  Crane overlap detection with LAG + running MAX
--   G  Materialise silver via COPY … TO PARQUET
-- =============================================================================


-- ─── A. Bronze views ──────────────────────────────────────────────────────────
-- One view per bronze table so every downstream CTE uses a stable name
-- rather than repeating the file path.

CREATE OR REPLACE VIEW bz_vessel_calls AS
    SELECT * FROM read_parquet('data/bronze/vessel_calls.parquet');

CREATE OR REPLACE VIEW bz_crane_assignments AS
    SELECT * FROM read_parquet('data/bronze/crane_assignments.parquet');

CREATE OR REPLACE VIEW bz_container_moves AS
    SELECT * FROM read_parquet('data/bronze/container_moves.parquet');

CREATE OR REPLACE VIEW bz_terminal_metadata AS
    SELECT * FROM read_parquet('data/bronze/terminal_metadata.parquet');

CREATE OR REPLACE VIEW bz_weather_daily AS
    SELECT * FROM read_parquet('data/bronze/weather_daily.parquet');


-- ─── B. Duplicate detection and classification ────────────────────────────────
-- Identify vessel_call_id values that appear more than once in bronze.
-- These originate from late re-transmissions by the source system.
-- Classifying the lag helps distinguish harmless retries from genuine
-- late corrections that may carry materially different timestamps or cargo.

WITH dup_audit AS (
    SELECT
        vessel_call_id,
        COUNT(*)                                                AS record_count,
        MIN(record_created_at)                                  AS earliest_record,
        MAX(record_created_at)                                  AS latest_record,
        datediff('hour',
                 MIN(record_created_at),
                 MAX(record_created_at))                        AS update_lag_hours
    FROM bz_vessel_calls
    GROUP BY vessel_call_id
    HAVING COUNT(*) > 1
)
SELECT
    vessel_call_id,
    record_count,
    earliest_record,
    latest_record,
    update_lag_hours,
    CASE
        WHEN update_lag_hours <  1  THEN 'system_retry'
        WHEN update_lag_hours < 24  THEN 'same_day_correction'
        ELSE                             'late_update'
    END AS duplicate_class
FROM dup_audit
ORDER BY update_lag_hours DESC;


-- ─── C. Latest-record deduplication ──────────────────────────────────────────
-- Keep exactly one row per vessel_call_id: the most recently written record.
--
-- QUALIFY is a DuckDB / Snowflake extension that filters on a window-function
-- result without a wrapping subquery.  It reads as: "keep only the row where
-- the row-number within each vessel_call_id partition equals 1."
--
-- When using QUALIFY, the window function is evaluated but not added to the
-- SELECT list, so SELECT * returns the original columns only — no EXCLUDE needed.

SELECT *
FROM bz_vessel_calls
QUALIFY
    ROW_NUMBER() OVER (
        PARTITION BY vessel_call_id
        ORDER BY
            record_created_at DESC NULLS LAST,
            _ingestion_ts     DESC NULLS LAST
    ) = 1;


-- ─── D. Invalid timestamp detection ──────────────────────────────────────────
-- Four structural timestamp problems are checked before the silver write.
-- Rows that fail any check are flagged; they stay in silver with boolean
-- flags so analysts can filter, count, and investigate them.

SELECT
    vessel_call_id,
    eta,
    ata,
    etd,
    atd,

    -- (1) Arrival recorded after departure — physically impossible.
    (ata IS NOT NULL AND atd IS NOT NULL AND ata > atd)
        AS invalid_arrival_sequence_flag,

    -- (2) Planned departure before planned arrival — scheduling entry error.
    (eta IS NOT NULL AND etd IS NOT NULL AND etd < eta)
        AS invalid_departure_sequence_flag,

    -- (3) Actual departure before actual arrival — same impossibility on actuals.
    (ata IS NOT NULL AND atd IS NOT NULL AND atd < ata)
        AS atd_before_ata_flag,

    -- (4) Turnaround exceeds seven days — operationally implausible; flag for review.
    (ata IS NOT NULL AND atd IS NOT NULL
     AND datediff('hour', ata, atd) > 168)
        AS excessive_turnaround_flag

FROM bz_vessel_calls
WHERE
       (ata IS NOT NULL AND atd IS NOT NULL AND ata > atd)
    OR (eta IS NOT NULL AND etd IS NOT NULL AND etd < eta)
    OR (ata IS NOT NULL AND atd IS NOT NULL
        AND datediff('hour', ata, atd) > 168);


-- ─── E. Delay metrics and quality flags ───────────────────────────────────────
-- One CTE per concern; the final SELECT assembles the full silver row.
--
-- The outlier threshold is computed once in a scalar CTE and broadcast to
-- every row via CROSS JOIN — avoids a correlated subquery on every row.

WITH

-- Step 1: deduplicate bronze
deduped AS (
    SELECT *
    FROM bz_vessel_calls
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY vessel_call_id
        ORDER BY record_created_at DESC NULLS LAST
    ) = 1
),

-- Step 2: derive signed delay in minutes
with_delays AS (
    SELECT
        *,
        CASE
            WHEN ata IS NOT NULL AND eta IS NOT NULL
            THEN datediff('second', eta, ata) / 60.0
        END AS arrival_delay_minutes,

        CASE
            WHEN atd IS NOT NULL AND etd IS NOT NULL
            THEN datediff('second', etd, atd) / 60.0
        END AS departure_delay_minutes
    FROM deduped
),

-- Step 3: Tukey upper fence on the distribution of positive delays.
--   fence = Q3 + 1.5 × IQR  (IQR = Q3 − Q1)
--   Floored at 24 h so isolated short delays never set a trivially low bar.
delay_fence AS (
    SELECT
        GREATEST(
            24.0,
            QUANTILE_CONT(delay_hours, 0.75)
                + 1.5 * (QUANTILE_CONT(delay_hours, 0.75)
                         - QUANTILE_CONT(delay_hours, 0.25))
        ) AS outlier_threshold_hours
    FROM with_delays
    WHERE delay_hours > 0
)

-- Step 4: attach quality flags and emit silver row
SELECT
    d.*,

    -- Missing-value flags
    (d.eta IS NULL)  AS missing_eta_flag,
    (d.ata IS NULL)  AS missing_ata_flag,

    -- Timestamp sequence flags
    (d.ata IS NOT NULL AND d.atd IS NOT NULL
     AND d.ata > d.atd)                                     AS invalid_arrival_sequence_flag,

    (d.eta IS NOT NULL AND d.etd IS NOT NULL
     AND d.etd < d.eta)                                      AS invalid_departure_sequence_flag,

    -- Statistical outlier: delay exceeds the Tukey fence
    (d.delay_hours IS NOT NULL
     AND d.delay_hours > f.outlier_threshold_hours)          AS large_delay_outlier_flag,

    NOW()                                                    AS _silver_ts

FROM with_delays d
CROSS JOIN delay_fence f;


-- ─── F. Crane overlap detection ───────────────────────────────────────────────
-- Within each crane's assignment sequence (sorted by actual_start), a row
-- overlaps a prior assignment when it begins before the highest actual_end
-- seen so far.  LAG gives the immediately preceding end; MAX OVER with a
-- bounded frame gives the true "high water mark" across all prior rows.
--
-- LEAD checks the opposite direction so the *earlier* party in an overlap
-- pair is also flagged, not just the later one.

WITH sorted_assignments AS (
    SELECT
        assignment_id,
        crane_id,
        vessel_call_id,
        actual_start,
        actual_end,

        -- Running maximum of actual_end across all prior rows for this crane.
        -- When actual_start < prior_max_end, this assignment overlaps at least
        -- one earlier assignment.
        MAX(actual_end) OVER (
            PARTITION BY crane_id
            ORDER BY actual_start
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_max_end,

        -- Next assignment's start for the same crane.
        -- When actual_end > next_actual_start, this assignment is the earlier
        -- party in the overlap pair.
        LEAD(actual_start) OVER (
            PARTITION BY crane_id
            ORDER BY actual_start
        ) AS next_actual_start

    FROM bz_crane_assignments
    WHERE actual_start IS NOT NULL
      AND actual_end   IS NOT NULL
      AND actual_end   >= actual_start     -- skip already-invalid rows
)
SELECT
    assignment_id,
    crane_id,
    actual_start,
    actual_end,
    prior_max_end,
    next_actual_start,
    (actual_start < prior_max_end OR actual_end > next_actual_start)
        AS crane_overlap_flag
FROM sorted_assignments
ORDER BY crane_id, actual_start;


-- ─── G. Materialise silver via COPY … TO ─────────────────────────────────────
-- DuckDB's COPY … TO writes a query result directly to Parquet without an
-- intermediate CREATE TABLE step.  Equivalent to a CTAS but keeps the storage
-- format explicit and independent of any attached database catalog.

COPY (
    WITH deduped AS (
        SELECT * EXCLUDE (rn)
        FROM bz_vessel_calls
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY vessel_call_id
            ORDER BY record_created_at DESC NULLS LAST
        ) = 1
    ),
    delay_fence AS (
        SELECT GREATEST(24.0,
            QUANTILE_CONT(delay_hours, 0.75)
                + 1.5 * (QUANTILE_CONT(delay_hours, 0.75)
                         - QUANTILE_CONT(delay_hours, 0.25))
        ) AS threshold
        FROM deduped
        WHERE delay_hours > 0
    )
    SELECT
        d.*,
        CASE WHEN d.ata IS NOT NULL AND d.eta IS NOT NULL
             THEN datediff('second', d.eta, d.ata) / 60.0 END  AS arrival_delay_minutes,
        CASE WHEN d.atd IS NOT NULL AND d.etd IS NOT NULL
             THEN datediff('second', d.etd, d.atd) / 60.0 END  AS departure_delay_minutes,
        (d.eta IS NULL)                                         AS missing_eta_flag,
        (d.ata IS NULL)                                         AS missing_ata_flag,
        (d.ata IS NOT NULL AND d.atd IS NOT NULL AND d.ata > d.atd)
                                                                AS invalid_arrival_sequence_flag,
        (d.delay_hours IS NOT NULL AND d.delay_hours > f.threshold)
                                                                AS large_delay_outlier_flag,
        NOW()                                                   AS _silver_ts
    FROM deduped d
    CROSS JOIN delay_fence f
)
TO 'data/silver/vessel_calls_silver_sql.parquet'
(FORMAT PARQUET, COMPRESSION SNAPPY);
