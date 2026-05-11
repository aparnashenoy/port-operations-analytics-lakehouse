-- =============================================================================
-- 05_business_analysis_queries.sql
-- Business intelligence queries for the port operations analytics lakehouse.
--
-- These queries are written for the operations and commercial teams and are
-- designed to answer specific business questions directly from the gold layer.
-- Each query is self-contained and carries a plain-English business question
-- as its header comment.
--
-- Queries
--   1   Monthly delay trend by terminal (MoM analysis)
--   2   Delay root cause attribution by terminal and carrier
--   3   Storm vs clear-weather performance comparison
--   4   Carrier performance scorecard
--   5   Vessel class throughput and efficiency analysis
--   6   Berth utilisation by terminal
--   7   Worst-performing days (operational incident detection)
--   8   Seasonal congestion heatmap
--   9   Crane productivity outlier identification
-- =============================================================================

CREATE OR REPLACE VIEW gold_daily_kpis AS
    SELECT * FROM read_parquet('data/gold/gold_daily_terminal_kpis.parquet');

CREATE OR REPLACE VIEW gold_call_summary AS
    SELECT * FROM read_parquet('data/gold/gold_vessel_call_summary.parquet');

CREATE OR REPLACE VIEW sv_vessel_calls AS
    SELECT * FROM read_parquet('data/silver/vessel_calls_silver.parquet');

CREATE OR REPLACE VIEW sv_crane_assignments AS
    SELECT * FROM read_parquet('data/silver/crane_assignments_silver.parquet');

CREATE OR REPLACE VIEW sv_terminal_metadata AS
    SELECT * FROM read_parquet('data/silver/terminal_metadata_silver.parquet');


-- ─── Query 1: Monthly delay trend by terminal ─────────────────────────────────
-- Business question: Is vessel delay getting better or worse at each terminal
-- over time?  Compare this month to the prior month to identify terminals
-- trending in the wrong direction before they become a commercial problem.

WITH monthly AS (
    SELECT
        strftime(operation_date, '%Y-%m')                               AS year_month,
        terminal_id,
        terminal_name,
        SUM(vessel_calls)                                               AS monthly_vessel_calls,
        ROUND(AVG(avg_arrival_delay_minutes), 1)                        AS avg_delay_min,
        ROUND(AVG(delayed_vessel_percentage), 1)                        AS avg_delayed_pct,
        SUM(storm_days_count)                                           AS storm_days
    FROM gold_daily_kpis
    GROUP BY year_month, terminal_id, terminal_name
),
with_mom AS (
    SELECT
        *,
        LAG(avg_delay_min) OVER (
            PARTITION BY terminal_id
            ORDER BY year_month
        )                                                               AS prior_month_delay_min,
        LAG(avg_delayed_pct) OVER (
            PARTITION BY terminal_id
            ORDER BY year_month
        )                                                               AS prior_month_delayed_pct
    FROM monthly
)
SELECT
    year_month,
    terminal_id,
    terminal_name,
    monthly_vessel_calls,
    avg_delay_min,
    prior_month_delay_min,
    ROUND(avg_delay_min - prior_month_delay_min, 1)                     AS delay_mom_delta_min,
    avg_delayed_pct,
    prior_month_delayed_pct,
    ROUND(avg_delayed_pct - prior_month_delayed_pct, 1)                 AS delayed_pct_mom_delta,
    storm_days,
    -- Annotate whether the month-over-month change is driven by weather
    CASE
        WHEN storm_days > 3 AND avg_delay_min > prior_month_delay_min
            THEN 'weather_driven_increase'
        WHEN avg_delay_min > prior_month_delay_min + 15
            THEN 'significant_deterioration'
        WHEN avg_delay_min < prior_month_delay_min - 15
            THEN 'significant_improvement'
        ELSE 'stable'
    END                                                                 AS trend_classification
FROM with_mom
WHERE prior_month_delay_min IS NOT NULL  -- skip first month (no prior)
ORDER BY year_month DESC, terminal_id;


-- ─── Query 2: Delay root cause attribution ────────────────────────────────────
-- Business question: What is causing delays, and does the mix of root causes
-- differ by terminal or carrier?  Knowing whether delays are weather-driven
-- or equipment-driven shapes whether the fix is operational or capital.

WITH delay_calls AS (
    SELECT
        terminal_id,
        carrier_code,
        delay_reason,
        delay_hours,
        arrival_delay_minutes,
        vessel_class
    FROM sv_vessel_calls
    WHERE delay_hours > 0
      AND atd IS NOT NULL
)
SELECT
    terminal_id,
    delay_reason,
    COUNT(*)                                                            AS call_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (
        PARTITION BY terminal_id
    ), 1)                                                               AS pct_of_terminal_delays,
    ROUND(AVG(delay_hours), 2)                                          AS avg_delay_hours,
    ROUND(QUANTILE_CONT(delay_hours, 0.50), 2)                         AS median_delay_hours,
    ROUND(QUANTILE_CONT(delay_hours, 0.95), 2)                         AS p95_delay_hours,
    -- Carriers most exposed to this reason
    STRING_AGG(DISTINCT carrier_code, ', ' ORDER BY carrier_code)
        FILTER (WHERE delay_hours > 5)                                  AS carriers_most_affected
FROM delay_calls
GROUP BY terminal_id, delay_reason
ORDER BY terminal_id, call_count DESC;


-- ─── Query 3: Storm vs clear-weather performance comparison ──────────────────
-- Business question: How much worse is terminal performance on storm days?
-- Quantify the weather penalty so it can be modelled in SLA discussions.

SELECT
    terminal_id,
    terminal_name,
    CASE WHEN storm_days_count = 1 THEN 'storm_day' ELSE 'clear_day' END
                                                                        AS weather_condition,
    COUNT(*)                                                            AS day_count,
    SUM(vessel_calls)                                                   AS total_vessel_calls,
    ROUND(AVG(avg_arrival_delay_minutes), 1)                           AS avg_arrival_delay_min,
    ROUND(AVG(delayed_vessel_percentage), 1)                           AS avg_delayed_vessel_pct,
    ROUND(AVG(avg_moves_per_crane_hour), 2)                            AS avg_crane_productivity
FROM gold_daily_kpis
GROUP BY terminal_id, terminal_name, weather_condition
ORDER BY terminal_id, weather_condition;

-- Summary: fleet-wide weather penalty in headline numbers
SELECT
    CASE WHEN storm_days_count = 1 THEN 'storm_day' ELSE 'clear_day' END
                                                                        AS weather_condition,
    COUNT(*)                                                            AS total_terminal_days,
    ROUND(AVG(avg_arrival_delay_minutes), 1)                           AS fleet_avg_delay_min,
    ROUND(AVG(delayed_vessel_percentage), 1)                           AS fleet_delayed_pct,
    ROUND(AVG(avg_moves_per_crane_hour), 2)                            AS fleet_crane_productivity
FROM gold_daily_kpis
GROUP BY weather_condition;


-- ─── Query 4: Carrier performance scorecard ───────────────────────────────────
-- Business question: Which shipping lines consistently arrive on time and
-- turn around quickly?  Poor performers may warrant SLA review or revised
-- slot allocation.
--
-- Avoiding double-count: aggregate from vessel_calls (one row per call),
-- not from gold_daily_kpis (which is already aggregated).

WITH carrier_stats AS (
    SELECT
        carrier_code,
        vessel_class,
        COUNT(*)                                                        AS total_calls,
        ROUND(AVG(arrival_delay_minutes), 1)                           AS avg_arrival_delay_min,
        ROUND(AVG(departure_delay_minutes), 1)                         AS avg_departure_delay_min,
        ROUND(AVG(actual_turnaround_hours), 2)                         AS avg_turnaround_hours,
        ROUND(AVG(planned_turnaround_hours), 2)                        AS avg_planned_turnaround_h,
        ROUND(100.0 * COUNT(*) FILTER (WHERE delay_hours > 0.5)
              / COUNT(*), 1)                                            AS delay_rate_pct,
        ROUND(QUANTILE_CONT(arrival_delay_minutes, 0.95), 1)          AS p95_arrival_delay_min,
        SUM(actual_cargo_teu)                                          AS total_actual_teu
    FROM sv_vessel_calls
    WHERE atd IS NOT NULL      -- completed calls only
      AND NOT invalid_arrival_sequence_flag
    GROUP BY carrier_code, vessel_class
),
ranked AS (
    SELECT
        *,
        RANK() OVER (ORDER BY delay_rate_pct ASC)                      AS punctuality_rank,
        RANK() OVER (ORDER BY avg_turnaround_hours ASC)                AS turnaround_rank,
        RANK() OVER (ORDER BY total_actual_teu DESC)                   AS volume_rank
    FROM carrier_stats
    WHERE total_calls >= 5     -- minimum sample size for meaningful comparison
)
SELECT
    carrier_code,
    vessel_class,
    total_calls,
    avg_arrival_delay_min,
    p95_arrival_delay_min,
    delay_rate_pct,
    avg_turnaround_hours,
    avg_planned_turnaround_h,
    ROUND(avg_turnaround_hours - avg_planned_turnaround_h, 2)          AS turnaround_variance_h,
    total_actual_teu,
    punctuality_rank,
    turnaround_rank,
    volume_rank
FROM ranked
ORDER BY delay_rate_pct ASC, total_calls DESC;


-- ─── Query 5: Vessel class throughput and efficiency ─────────────────────────
-- Business question: Do larger vessels use port resources proportionally
-- more efficiently, or do they create disproportionate congestion?
-- Helps inform slot pricing and berth assignment policy.

SELECT
    vc.vessel_class,
    COUNT(DISTINCT vc.vessel_call_id)                                   AS total_calls,
    ROUND(AVG(vc.planned_turnaround_hours), 1)                         AS avg_planned_turnaround_h,
    ROUND(AVG(vc.actual_turnaround_hours), 1)                          AS avg_actual_turnaround_h,
    ROUND(AVG(vc.actual_turnaround_hours - vc.planned_turnaround_hours), 2)
                                                                        AS avg_turnaround_overrun_h,
    ROUND(AVG(vc.actual_cargo_teu), 0)                                 AS avg_teu_per_call,
    SUM(vc.actual_cargo_teu)                                           AS total_teu,
    -- TEU per hour: measures how efficiently each class uses berth time
    ROUND(SUM(vc.actual_cargo_teu)
          / NULLIF(SUM(vc.actual_turnaround_hours), 0), 1)             AS teu_per_turnaround_hour,
    ROUND(100.0 * COUNT(*) FILTER (WHERE vc.delay_hours > 0.5)
          / COUNT(*), 1)                                                AS delay_rate_pct,
    -- Crane demand: average crane hours consumed per call
    ROUND(AVG(cs.total_crane_hours), 2)                                AS avg_crane_hours_per_call
FROM sv_vessel_calls vc
LEFT JOIN gold_call_summary cs USING (vessel_call_id)
WHERE vc.atd IS NOT NULL
  AND NOT vc.invalid_arrival_sequence_flag
GROUP BY vc.vessel_class
ORDER BY
    CASE vc.vessel_class
        WHEN 'Feeder'       THEN 1
        WHEN 'Sub-Panamax'  THEN 2
        WHEN 'Panamax'      THEN 3
        WHEN 'Post-Panamax' THEN 4
        WHEN 'ULCV'         THEN 5
    END;


-- ─── Query 6: Berth utilisation by terminal ───────────────────────────────────
-- Business question: How close to capacity are terminals running?
-- A utilisation rate above 85% signals that a single storm or large call
-- can cascade into systemic delays.
--
-- Berth-hours available = berth_count × 24 h × days in period.
-- Berth-hours consumed  = sum of actual turnaround hours across all calls.

WITH period_bounds AS (
    SELECT
        MIN(date_trunc('day', ata))  AS period_start,
        MAX(date_trunc('day', atd))  AS period_end,
        datediff('day',
                 MIN(date_trunc('day', ata)),
                 MAX(date_trunc('day', atd))) + 1  AS period_days
    FROM sv_vessel_calls
    WHERE ata IS NOT NULL AND atd IS NOT NULL
),
berth_consumption AS (
    SELECT
        vc.terminal_id,
        SUM(vc.actual_turnaround_hours)         AS consumed_berth_hours,
        COUNT(DISTINCT vc.vessel_call_id)        AS vessel_calls
    FROM sv_vessel_calls vc
    WHERE vc.ata IS NOT NULL AND vc.atd IS NOT NULL
    GROUP BY vc.terminal_id
)
SELECT
    bc.terminal_id,
    tm.terminal_name,
    tm.berth_count,
    bc.vessel_calls,
    ROUND(bc.consumed_berth_hours, 1)                                   AS consumed_berth_hours,
    -- Available capacity assumes 24h / day / berth with 85% operational uptime
    ROUND(tm.berth_count * p.period_days * 24.0 * 0.85, 0)            AS available_berth_hours,
    ROUND(100.0 * bc.consumed_berth_hours
          / (tm.berth_count * p.period_days * 24.0 * 0.85), 1)        AS berth_utilisation_pct,
    CASE
        WHEN bc.consumed_berth_hours / (tm.berth_count * p.period_days * 24.0 * 0.85)
             > 0.85 THEN 'CRITICAL — over 85% utilised'
        WHEN bc.consumed_berth_hours / (tm.berth_count * p.period_days * 24.0 * 0.85)
             > 0.70 THEN 'HIGH — monitor closely'
        ELSE             'NORMAL'
    END                                                                 AS utilisation_alert
FROM berth_consumption bc
JOIN sv_terminal_metadata tm USING (terminal_id)
CROSS JOIN period_bounds p
ORDER BY berth_utilisation_pct DESC;


-- ─── Query 7: Worst-performing days (incident detection) ──────────────────────
-- Business question: Which specific terminal-days had the worst delay
-- performance?  These are candidates for post-incident review and are
-- the clearest test of whether operational changes are working.

SELECT
    operation_date::DATE                                                AS operation_date,
    terminal_id,
    terminal_name,
    vessel_calls,
    ROUND(avg_arrival_delay_minutes, 1)                                AS avg_arrival_delay_min,
    ROUND(delayed_vessel_percentage, 1)                                AS delayed_vessel_pct,
    data_quality_issue_count,
    storm_days_count,
    -- Score: higher is worse (weighted sum of delay signal)
    ROUND(
        avg_arrival_delay_minutes * 0.4
        + delayed_vessel_percentage * 2.0
        + data_quality_issue_count * 5.0
        + storm_days_count * 10.0,
    1)                                                                  AS incident_score
FROM gold_daily_kpis
WHERE vessel_calls >= 2     -- single-call days have high noise
ORDER BY incident_score DESC
LIMIT 20;


-- ─── Query 8: Seasonal congestion heatmap ────────────────────────────────────
-- Business question: Which months and days of the week see peak traffic
-- and longest delays?  Useful for headcount planning and slot allocation.
--
-- Result is suited for a pivot / heatmap visualisation in a BI tool.

SELECT
    EXTRACT(month  FROM operation_date)                                 AS month_num,
    strftime(operation_date, '%b')                                      AS month_name,
    EXTRACT(isodow FROM operation_date)                                 AS dow_num,
    CASE EXTRACT(isodow FROM operation_date)
        WHEN 1 THEN 'Mon' WHEN 2 THEN 'Tue' WHEN 3 THEN 'Wed'
        WHEN 4 THEN 'Thu' WHEN 5 THEN 'Fri' WHEN 6 THEN 'Sat'
        WHEN 7 THEN 'Sun'
    END                                                                 AS dow_name,
    ROUND(AVG(vessel_calls), 2)                                         AS avg_daily_vessel_calls,
    ROUND(AVG(avg_arrival_delay_minutes), 1)                           AS avg_delay_min,
    ROUND(AVG(delayed_vessel_percentage), 1)                           AS avg_delayed_pct,
    COUNT(DISTINCT operation_date::DATE)                                AS sample_days
FROM gold_daily_kpis
GROUP BY month_num, month_name, dow_num, dow_name
ORDER BY month_num, dow_num;


-- ─── Query 9: Crane productivity outlier identification ───────────────────────
-- Business question: Which specific crane-vessel combinations produced
-- significantly below-average throughput?  Persistent underperformance by
-- a single crane suggests a maintenance issue.

WITH crane_stats AS (
    SELECT
        terminal_id,
        crane_id,
        COUNT(*)                                                        AS assignment_count,
        ROUND(AVG(productivity_moves_per_hour), 2)                     AS avg_productivity,
        ROUND(STDDEV(productivity_moves_per_hour), 2)                  AS stddev_productivity,
        ROUND(MIN(productivity_moves_per_hour), 2)                     AS min_productivity,
        ROUND(MAX(productivity_moves_per_hour), 2)                     AS max_productivity,
        ROUND(SUM(actual_crane_hours), 1)                              AS total_hours_worked
    FROM sv_crane_assignments
    WHERE NOT invalid_crane_time_flag
      AND actual_crane_hours > 0
    GROUP BY terminal_id, crane_id
),
terminal_benchmarks AS (
    SELECT
        terminal_id,
        ROUND(AVG(productivity_moves_per_hour), 2)                     AS terminal_avg_productivity,
        ROUND(STDDEV(productivity_moves_per_hour), 2)                  AS terminal_stddev
    FROM sv_crane_assignments
    WHERE NOT invalid_crane_time_flag
    GROUP BY terminal_id
)
SELECT
    cs.terminal_id,
    cs.crane_id,
    cs.assignment_count,
    cs.avg_productivity,
    tb.terminal_avg_productivity,
    ROUND(cs.avg_productivity - tb.terminal_avg_productivity, 2)       AS productivity_vs_benchmark,
    -- Z-score: how many standard deviations below the terminal mean
    ROUND((cs.avg_productivity - tb.terminal_avg_productivity)
          / NULLIF(tb.terminal_stddev, 0), 2)                          AS z_score,
    cs.total_hours_worked,
    CASE
        WHEN cs.avg_productivity < tb.terminal_avg_productivity - 2 * tb.terminal_stddev
            THEN 'UNDERPERFORMING — review maintenance log'
        WHEN cs.avg_productivity > tb.terminal_avg_productivity + 2 * tb.terminal_stddev
            THEN 'OUTPERFORMING — validate data accuracy'
        ELSE 'NORMAL'
    END                                                                 AS performance_flag
FROM crane_stats cs
JOIN terminal_benchmarks tb USING (terminal_id)
ORDER BY z_score ASC;
