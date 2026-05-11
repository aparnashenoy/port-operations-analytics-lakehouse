-- =============================================================================
-- 02_gold_terminal_kpis.sql
-- Gold-layer KPI aggregation expressed in DuckDB SQL.
--
-- Demonstrates:
--   • Pre-aggregation CTEs to avoid duplicate counts after multi-table joins
--   • FILTER (WHERE …) for conditional aggregation without nested CASE
--   • RANGE BETWEEN INTERVAL for time-windowed rolling averages
--   • RANK / DENSE_RANK for terminal performance league tables
--   • GROUPING SETS for simultaneous fleet-wide and per-terminal totals
--   • Source freshness monitoring via _ingestion_ts lag
-- =============================================================================


-- ─── 0. Source views ──────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW sv_vessel_calls AS
    SELECT * FROM read_parquet('data/silver/vessel_calls_silver.parquet');

CREATE OR REPLACE VIEW sv_crane_assignments AS
    SELECT * FROM read_parquet('data/silver/crane_assignments_silver.parquet');

CREATE OR REPLACE VIEW sv_container_moves AS
    SELECT * FROM read_parquet('data/silver/container_moves_silver.parquet');

CREATE OR REPLACE VIEW sv_terminal_metadata AS
    SELECT * FROM read_parquet('data/silver/terminal_metadata_silver.parquet');

CREATE OR REPLACE VIEW sv_weather_daily AS
    SELECT * FROM read_parquet('data/silver/weather_daily_silver.parquet');


-- ─── 1. Daily terminal KPI table ─────────────────────────────────────────────
-- Pre-aggregate moves and crane hours to vessel_call_id before joining to
-- vessel_calls.  Joining the raw tables first and then aggregating causes
-- fan-out: one vessel call with N crane rows and M move rows produces N×M
-- intermediate rows, inflating all counts.

WITH

-- 1a. Aggregate completed moves to call level
moves_per_call AS (
    SELECT
        vessel_call_id,
        COUNT(*) FILTER (WHERE move_status = 'completed') AS actual_moves,
        COUNT(*)                                           AS planned_moves
    FROM sv_container_moves
    GROUP BY vessel_call_id
),

-- 1b. Aggregate valid crane hours to call level.
-- Clamp actual_crane_hours to zero to prevent inverted-timestamp rows from
-- deflating totals; exclude fully invalid assignments from utilisation hours.
crane_per_call AS (
    SELECT
        vessel_call_id,
        SUM(GREATEST(actual_crane_hours, 0))                               AS total_crane_hours,
        SUM(GREATEST(actual_crane_hours, 0))
            FILTER (WHERE NOT invalid_crane_time_flag)                     AS valid_crane_hours,
        COUNT(DISTINCT crane_id)                                           AS cranes_deployed,
        SUM(CAST(missing_crane_id_flag   AS INTEGER)
          + CAST(invalid_crane_time_flag  AS INTEGER)
          + CAST(crane_overlap_flag       AS INTEGER))                     AS crane_dq_flags
    FROM sv_crane_assignments
    GROUP BY vessel_call_id
),

-- 1c. Derive operation_date (ATA if known, else ETA) and sum all DQ flags
enriched_calls AS (
    SELECT
        vc.vessel_call_id,
        vc.terminal_id,
        COALESCE(date_trunc('day', vc.ata),
                 date_trunc('day', vc.eta))                                AS operation_date,
        vc.arrival_delay_minutes,
        vc.departure_delay_minutes,
        vc.delay_hours,
        vc.weather_impact_factor,
        -- A call is "delayed" when it arrived more than 30 minutes late
        (vc.arrival_delay_minutes > 30)                                    AS is_delayed,
        COALESCE(m.actual_moves, 0)                                        AS actual_moves,
        COALESCE(c.total_crane_hours, 0)                                   AS total_crane_hours,
        COALESCE(c.valid_crane_hours,  0)                                  AS valid_crane_hours,
        COALESCE(c.crane_dq_flags,     0)
          + CAST(vc.missing_eta_flag             AS INTEGER)
          + CAST(vc.missing_ata_flag             AS INTEGER)
          + CAST(vc.invalid_arrival_sequence_flag AS INTEGER)
          + CAST(vc.large_delay_outlier_flag      AS INTEGER)              AS total_dq_flags
    FROM sv_vessel_calls vc
    LEFT JOIN moves_per_call m USING (vessel_call_id)
    LEFT JOIN crane_per_call  c USING (vessel_call_id)
    WHERE vc.ata IS NOT NULL OR vc.eta IS NOT NULL  -- drop calls with no date anchor
),

-- 1d. Join weather at terminal-day grain (one severity reading per day per terminal)
with_weather AS (
    SELECT
        e.*,
        w.severity_index,
        (w.severity_index > 0.5) AS is_storm_day
    FROM enriched_calls e
    LEFT JOIN sv_weather_daily w
           ON  w.terminal_id  = e.terminal_id
           AND date_trunc('day', w.weather_date) = e.operation_date
),

-- 1e. Join terminal name (reference join — always one row per terminal_id)
with_terminal AS (
    SELECT
        ww.*,
        tm.terminal_name
    FROM with_weather ww
    LEFT JOIN sv_terminal_metadata tm USING (terminal_id)
)

-- 1f. Aggregate to terminal-day grain
SELECT
    operation_date,
    terminal_id,
    terminal_name,
    COUNT(*)                                                            AS vessel_calls,
    ROUND(AVG(arrival_delay_minutes), 1)                               AS avg_arrival_delay_minutes,
    ROUND(AVG(departure_delay_minutes), 1)                             AS avg_departure_delay_minutes,
    ROUND(100.0 * SUM(CAST(is_delayed AS INTEGER)) / COUNT(*), 1)     AS delayed_vessel_pct,
    SUM(actual_moves)                                                  AS total_container_moves,
    ROUND(SUM(total_crane_hours), 2)                                   AS total_crane_hours,
    ROUND(SUM(valid_crane_hours),  2)                                  AS crane_utilization_hours,
    -- Pooled moves-per-crane-hour avoids division-by-zero at call level
    ROUND(SUM(actual_moves)
          / NULLIF(SUM(valid_crane_hours), 0), 2)                     AS avg_moves_per_crane_hour,
    SUM(total_dq_flags)                                                AS data_quality_issue_count,
    MAX(CAST(is_storm_day AS INTEGER))                                 AS storm_days_count
FROM with_terminal
GROUP BY operation_date, terminal_id, terminal_name
ORDER BY operation_date, terminal_id;


-- ─── 2. Seven-day rolling average delay ───────────────────────────────────────
-- RANGE BETWEEN INTERVAL works with DATE / TIMESTAMP ORDER BY keys in DuckDB.
-- The window spans a full week including the current day: [day-6 … day].
-- Useful for smoothing day-to-day noise in an operational dashboard.

WITH daily_kpis AS (
    -- Reference the gold Parquet directly; substitute the CTE above
    -- when running without the materialised gold table.
    SELECT
        operation_date,
        terminal_id,
        terminal_name,
        avg_arrival_delay_minutes,
        delayed_vessel_percentage
    FROM read_parquet('data/gold/gold_daily_terminal_kpis.parquet')
)
SELECT
    operation_date,
    terminal_id,
    terminal_name,
    ROUND(avg_arrival_delay_minutes, 1)                              AS daily_avg_delay_min,
    ROUND(AVG(avg_arrival_delay_minutes) OVER (
        PARTITION BY terminal_id
        ORDER BY operation_date
        RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW
    ), 1)                                                            AS rolling_7d_avg_delay_min,
    ROUND(AVG(delayed_vessel_percentage) OVER (
        PARTITION BY terminal_id
        ORDER BY operation_date
        RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW
    ), 1)                                                            AS rolling_7d_delayed_pct
FROM daily_kpis
ORDER BY terminal_id, operation_date;


-- ─── 3. Terminal performance ranking ─────────────────────────────────────────
-- RANK assigns the same rank to ties and skips the next rank value.
-- DENSE_RANK never skips — use it when gaps in the ranking would confuse
-- stakeholders ("why is there no rank 3?").
--
-- The subquery first aggregates to terminal level over the full period so the
-- ranking reflects overall performance, not a single day.

WITH terminal_summary AS (
    SELECT
        terminal_id,
        terminal_name,
        SUM(vessel_calls)                                            AS total_vessel_calls,
        ROUND(AVG(avg_arrival_delay_minutes), 1)                    AS period_avg_delay_min,
        ROUND(AVG(delayed_vessel_percentage), 1)                    AS period_delayed_pct,
        ROUND(SUM(total_crane_hours), 0)                            AS total_crane_hours,
        ROUND(AVG(avg_moves_per_crane_hour), 2)                     AS avg_productivity
    FROM read_parquet('data/gold/gold_daily_terminal_kpis.parquet')
    GROUP BY terminal_id, terminal_name
)
SELECT
    terminal_id,
    terminal_name,
    total_vessel_calls,
    period_avg_delay_min,
    period_delayed_pct,
    total_crane_hours,
    avg_productivity,

    -- Lower delay = better performance → rank ascending on delay
    RANK()        OVER (ORDER BY period_avg_delay_min ASC)          AS delay_rank,
    -- Higher productivity = better → rank descending
    DENSE_RANK()  OVER (ORDER BY avg_productivity DESC)             AS productivity_rank
FROM terminal_summary
ORDER BY delay_rank;


-- ─── 4. Fleet-wide and per-terminal totals (GROUPING SETS) ───────────────────
-- GROUPING SETS computes multiple aggregation levels in a single scan.
-- Here we get both per-terminal totals and a fleet-wide rollup row.
-- GROUPING(terminal_id) = 1 marks the rollup row so it can be styled
-- differently in a BI tool.

SELECT
    CASE WHEN GROUPING(terminal_id) = 1 THEN 'ALL TERMINALS'
         ELSE terminal_id END                                        AS terminal_id,
    CASE WHEN GROUPING(terminal_id) = 1 THEN 'Fleet Total'
         ELSE MAX(terminal_name) END                                 AS terminal_name,
    SUM(vessel_calls)                                                AS total_vessel_calls,
    ROUND(AVG(avg_arrival_delay_minutes), 1)                        AS avg_arrival_delay_min,
    ROUND(SUM(total_container_moves), 0)                            AS total_moves,
    ROUND(SUM(total_crane_hours), 1)                                AS total_crane_hours,
    GROUPING(terminal_id)                                            AS is_rollup_row
FROM read_parquet('data/gold/gold_daily_terminal_kpis.parquet')
GROUP BY GROUPING SETS (
    (terminal_id),   -- one row per terminal
    ()               -- one fleet-wide rollup row
)
ORDER BY is_rollup_row, terminal_id;


-- ─── 5. Source freshness monitoring ──────────────────────────────────────────
-- Stale data is an invisible operational risk.  This query computes the lag
-- between the latest ingestion timestamp and now for every bronze table,
-- raising a flag when any table has not been refreshed within 48 hours.
-- Run as a scheduled data-quality check or a pre-flight assertion.

WITH source_freshness AS (
    SELECT 'vessel_calls'      AS table_name,
           MAX(_ingestion_ts)  AS latest_ingestion
    FROM read_parquet('data/bronze/vessel_calls.parquet')

    UNION ALL BY NAME

    SELECT 'crane_assignments' AS table_name,
           MAX(_ingestion_ts)  AS latest_ingestion
    FROM read_parquet('data/bronze/crane_assignments.parquet')

    UNION ALL BY NAME

    SELECT 'container_moves'   AS table_name,
           MAX(_ingestion_ts)  AS latest_ingestion
    FROM read_parquet('data/bronze/container_moves.parquet')

    UNION ALL BY NAME

    SELECT 'weather_daily'     AS table_name,
           MAX(_ingestion_ts)  AS latest_ingestion
    FROM read_parquet('data/bronze/weather_daily.parquet')
)
SELECT
    table_name,
    latest_ingestion,
    ROUND(datediff('hour',
                   latest_ingestion::TIMESTAMP,
                   NOW()), 1)                          AS hours_since_ingestion,
    CASE
        WHEN datediff('hour',
                      latest_ingestion::TIMESTAMP,
                      NOW()) > 48 THEN 'STALE ⚠'
        ELSE 'OK'
    END                                                AS freshness_status
FROM source_freshness
ORDER BY hours_since_ingestion DESC;
