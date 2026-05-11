-- =============================================================================
-- 04_ml_features.sql
-- Point-in-time correct ML feature set for vessel delay prediction.
--
-- Business goal: predict whether a vessel call will be delayed by more than
-- 30 minutes at the time of ETA notification, using only information that
-- would realistically be available to an operations analyst at that moment.
--
-- Point-in-time (PIT) correctness rule:
--   Features derived from historical data must use only records whose
--   atd (actual time of departure) is strictly before the ETA of the
--   call being predicted.  Leaking future actuals causes optimistic models
--   that fail in production.
--
-- Feature groups produced
--   A  Vessel-level history (delay rate, avg delay, turnaround trend)
--   B  Terminal-level history (congestion, delay rate, productivity)
--   C  Berth congestion (concurrent calls at ETA)
--   D  Weather at ETA
--   E  Temporal / seasonal
--   F  Label column (is_delayed: delay_hours > 0.5)
-- =============================================================================

CREATE OR REPLACE VIEW sv_vessel_calls AS
    SELECT * FROM read_parquet('data/silver/vessel_calls_silver.parquet');

CREATE OR REPLACE VIEW sv_weather_daily AS
    SELECT * FROM read_parquet('data/silver/weather_daily_silver.parquet');

CREATE OR REPLACE VIEW sv_crane_assignments AS
    SELECT * FROM read_parquet('data/silver/crane_assignments_silver.parquet');


-- ─── Candidate calls ─────────────────────────────────────────────────────────
-- Only calls with a known ETA can be prediction targets.
-- Calls still in progress (atd IS NULL) are excluded from training data
-- because the label is unknown; they are valid inference targets at runtime.

CREATE OR REPLACE VIEW prediction_candidates AS
SELECT
    vessel_call_id,
    vessel_id,
    vessel_class,
    carrier_code,
    terminal_id,
    berth_id,
    eta,
    ata,
    atd,
    delay_hours,
    delay_reason,
    planned_turnaround_hours,
    actual_turnaround_hours,
    weather_impact_factor,
    missing_eta_flag,
    invalid_arrival_sequence_flag
FROM sv_vessel_calls
WHERE eta IS NOT NULL
  AND NOT invalid_arrival_sequence_flag   -- exclude records with bad timestamps
  AND NOT missing_eta_flag;


-- ─── A. Vessel-level historical features ─────────────────────────────────────
-- For each candidate call, compute statistics over all *prior completed calls*
-- for the same vessel.  "Prior" means atd < current call's ETA.
--
-- This self-join is the canonical point-in-time join pattern.  The LEFT join
-- ensures vessels with no prior history still appear (with NULL features).

WITH vessel_history AS (
    SELECT
        future.vessel_call_id                                           AS vessel_call_id,
        future.vessel_id,
        future.eta                                                      AS prediction_as_of,

        -- How often has this vessel been delayed historically?
        COUNT(past.vessel_call_id)                                      AS vessel_prior_call_count,
        ROUND(AVG(CASE WHEN past.delay_hours > 0.5 THEN 1.0 ELSE 0.0 END), 4)
                                                                        AS vessel_historical_delay_rate,
        ROUND(AVG(past.delay_hours), 2)                                 AS vessel_avg_delay_hours,

        -- Trend: compare last-3-call avg to overall avg.
        -- Positive trend_delta means the vessel has been getting worse recently.
        ROUND(
            AVG(past.delay_hours) FILTER (WHERE past.rn_desc <= 3)
            - AVG(past.delay_hours),
        2)                                                              AS vessel_delay_trend_delta,

        ROUND(AVG(past.actual_turnaround_hours), 2)                    AS vessel_avg_turnaround_hours,
        ROUND(STDDEV(past.actual_turnaround_hours), 2)                 AS vessel_turnaround_stddev,
        MAX(past.atd)                                                   AS vessel_last_call_atd

    FROM prediction_candidates future
    -- Point-in-time join: only prior completed calls for the same vessel
    LEFT JOIN (
        SELECT
            vessel_call_id,
            vessel_id,
            atd,
            delay_hours,
            actual_turnaround_hours,
            ROW_NUMBER() OVER (
                PARTITION BY vessel_id
                ORDER BY atd DESC
            ) AS rn_desc
        FROM sv_vessel_calls
        WHERE atd IS NOT NULL
          AND status = 'completed'
    ) past
        ON  past.vessel_id = future.vessel_id
        AND past.atd < future.eta      -- strict: no future leakage

    GROUP BY future.vessel_call_id, future.vessel_id, future.eta
)
SELECT * FROM vessel_history;


-- ─── B. Terminal-level historical features ────────────────────────────────────
-- Compute the terminal's delay environment over the 90 days preceding the
-- candidate call's ETA.  A busy, high-delay terminal is a strong predictor.

WITH terminal_history AS (
    SELECT
        future.vessel_call_id,
        future.terminal_id,
        future.eta AS prediction_as_of,

        COUNT(past.vessel_call_id)                                      AS terminal_90d_call_count,
        ROUND(AVG(CASE WHEN past.delay_hours > 0.5 THEN 1.0 ELSE 0.0 END), 4)
                                                                        AS terminal_90d_delay_rate,
        ROUND(AVG(past.delay_hours), 2)                                 AS terminal_90d_avg_delay_hours,
        ROUND(AVG(past.actual_turnaround_hours), 2)                     AS terminal_90d_avg_turnaround,

        -- Congestion index: avg concurrent calls at this terminal over the window.
        -- Busier terminal → longer waits for berths and cranes.
        ROUND(COUNT(past.vessel_call_id) / 90.0, 3)                    AS terminal_avg_daily_calls

    FROM prediction_candidates future
    LEFT JOIN sv_vessel_calls past
        ON  past.terminal_id = future.terminal_id
        AND past.atd IS NOT NULL
        AND past.atd < future.eta
        AND past.atd >= future.eta - INTERVAL 90 DAYS  -- 90-day look-back window

    GROUP BY future.vessel_call_id, future.terminal_id, future.eta
)
SELECT * FROM terminal_history;


-- ─── C. Berth congestion index at ETA ────────────────────────────────────────
-- Count how many other calls are scheduled to be *in berth* (ata ≤ ETA ≤ atd)
-- at the same terminal when the candidate vessel is due to arrive.
-- High concurrent occupancy creates queueing delays.
--
-- LATERAL join formulation: for each candidate, run a correlated aggregate.
-- DuckDB supports LATERAL joins with FROM … , LATERAL (SELECT ...) syntax.

SELECT
    c.vessel_call_id,
    c.terminal_id,
    c.eta,
    berth.concurrent_calls,
    berth.concurrent_calls_same_class
FROM prediction_candidates c,
LATERAL (
    SELECT
        COUNT(*)                                                        AS concurrent_calls,
        COUNT(*) FILTER (WHERE other.vessel_class = c.vessel_class)    AS concurrent_calls_same_class
    FROM sv_vessel_calls other
    WHERE other.terminal_id = c.terminal_id
      AND other.vessel_call_id <> c.vessel_call_id
      AND other.eta IS NOT NULL
      AND other.atd IS NOT NULL
      -- The other call overlaps the candidate's ETA
      AND other.eta <= c.eta
      AND other.atd >= c.eta
) berth
ORDER BY concurrent_calls DESC;


-- ─── D. Weather features at ETA ───────────────────────────────────────────────
-- Join to the weather reading for the terminal on the day of ETA.
-- Weather is a leading indicator of delay: high severity → pilot delays,
-- berth congestion, crane stoppages.

SELECT
    c.vessel_call_id,
    c.terminal_id,
    date_trunc('day', c.eta)                                            AS eta_date,
    w.wind_speed_knots,
    w.wave_height_m,
    w.visibility_nm,
    w.severity_index,
    w.weather_impact_factor,
    (w.severity_index > 0.5)                                            AS storm_at_eta_flag,
    -- Preceding-day severity can indicate a lingering backlog
    LAG(w.severity_index) OVER (
        PARTITION BY w.terminal_id
        ORDER BY w.weather_date
    )                                                                   AS prev_day_severity_index
FROM prediction_candidates c
LEFT JOIN sv_weather_daily w
       ON  w.terminal_id  = c.terminal_id
       AND date_trunc('day', w.weather_date) = date_trunc('day', c.eta);


-- ─── E. Temporal / seasonal features ─────────────────────────────────────────
-- Calendar signals capture systematic patterns that persist across vessels
-- and terminals: weekends have different pilot availability, winter has
-- worse weather, month-end has cargo surges.

SELECT
    c.vessel_call_id,
    c.eta,

    EXTRACT(month       FROM c.eta)                        AS eta_month,
    EXTRACT(isodow      FROM c.eta)                        AS eta_day_of_week,
    EXTRACT(hour        FROM c.eta)                        AS eta_hour,
    (EXTRACT(isodow FROM c.eta) IN (6, 7))                 AS is_weekend,

    CASE EXTRACT(month FROM c.eta)
        WHEN 12 THEN 'Q4' WHEN 1 THEN 'Q1' WHEN 2 THEN 'Q1' WHEN 3  THEN 'Q1'
        WHEN 4  THEN 'Q2' WHEN 5 THEN 'Q2' WHEN 6 THEN 'Q2' WHEN 7  THEN 'Q3'
        WHEN 8  THEN 'Q3' WHEN 9 THEN 'Q3' WHEN 10 THEN 'Q4' WHEN 11 THEN 'Q4'
    END                                                    AS eta_quarter,

    (EXTRACT(hour FROM c.eta) >= 20 OR EXTRACT(hour FROM c.eta) < 6)
                                                           AS is_night_arrival,

    -- Days since vessel's previous call at this port (loyalty/familiarity proxy)
    datediff('day',
             MAX(past.atd) OVER (
                 PARTITION BY c.vessel_id
                 ORDER BY c.eta
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
             ),
             c.eta)                                        AS days_since_last_call

FROM prediction_candidates c
LEFT JOIN sv_vessel_calls past
       ON  past.vessel_id = c.vessel_id
       AND past.atd < c.eta;


-- ─── F. Assembled feature table with label ────────────────────────────────────
-- Join all feature groups into a single flat table ready for model training.
-- Label: is_delayed = 1 when delay_hours > 0.5 (30 minutes, meaningful delay).
--
-- This query demonstrates how to build a training set without data leakage:
-- every feature is a function of information available at ETA time.

WITH

vessel_feats AS (
    SELECT
        future.vessel_call_id,
        COUNT(past.vessel_call_id)                                             AS vessel_prior_calls,
        ROUND(AVG(CASE WHEN past.delay_hours > 0.5 THEN 1.0 ELSE 0.0 END), 4) AS vessel_delay_rate,
        ROUND(AVG(past.delay_hours), 2)                                        AS vessel_avg_delay_h,
        ROUND(AVG(past.actual_turnaround_hours), 2)                            AS vessel_avg_turnaround_h
    FROM prediction_candidates future
    LEFT JOIN sv_vessel_calls past
           ON  past.vessel_id = future.vessel_id
           AND past.atd IS NOT NULL AND past.atd < future.eta
    GROUP BY future.vessel_call_id
),

terminal_feats AS (
    SELECT
        future.vessel_call_id,
        COUNT(past.vessel_call_id)                                             AS terminal_90d_calls,
        ROUND(AVG(CASE WHEN past.delay_hours > 0.5 THEN 1.0 ELSE 0.0 END), 4) AS terminal_delay_rate,
        ROUND(AVG(past.actual_turnaround_hours), 2)                            AS terminal_avg_turnaround_h
    FROM prediction_candidates future
    LEFT JOIN sv_vessel_calls past
           ON  past.terminal_id = future.terminal_id
           AND past.atd IS NOT NULL
           AND past.atd  < future.eta
           AND past.atd >= future.eta - INTERVAL 90 DAYS
    GROUP BY future.vessel_call_id
),

berth_feats AS (
    SELECT
        c.vessel_call_id,
        COUNT(other.vessel_call_id)                                            AS concurrent_berth_calls
    FROM prediction_candidates c
    LEFT JOIN sv_vessel_calls other
           ON  other.terminal_id        = c.terminal_id
           AND other.vessel_call_id    <> c.vessel_call_id
           AND other.eta               <= c.eta
           AND other.atd               >= c.eta
    GROUP BY c.vessel_call_id
),

weather_feats AS (
    SELECT
        c.vessel_call_id,
        COALESCE(w.severity_index,        0)                                   AS weather_severity,
        COALESCE(w.wind_speed_knots,      0)                                   AS wind_speed_knots,
        COALESCE(w.weather_impact_factor, 1)                                   AS weather_impact_factor,
        COALESCE(w.severity_index > 0.5, FALSE)                                AS storm_at_eta
    FROM prediction_candidates c
    LEFT JOIN sv_weather_daily w
           ON  w.terminal_id = c.terminal_id
           AND date_trunc('day', w.weather_date) = date_trunc('day', c.eta)
)

SELECT
    -- Identifiers (drop before model training)
    c.vessel_call_id,
    c.eta,

    -- Vessel features
    c.vessel_class,
    c.carrier_code,
    c.planned_turnaround_hours,
    COALESCE(vf.vessel_prior_calls,    0)                     AS vessel_prior_calls,
    COALESCE(vf.vessel_delay_rate,     0)                     AS vessel_delay_rate,
    COALESCE(vf.vessel_avg_delay_h,    0)                     AS vessel_avg_delay_h,
    COALESCE(vf.vessel_avg_turnaround_h, c.planned_turnaround_hours)
                                                              AS vessel_avg_turnaround_h,

    -- Terminal features
    COALESCE(tf.terminal_90d_calls,       0)                  AS terminal_90d_calls,
    COALESCE(tf.terminal_delay_rate,      0)                  AS terminal_delay_rate,
    COALESCE(tf.terminal_avg_turnaround_h, c.planned_turnaround_hours)
                                                              AS terminal_avg_turnaround_h,

    -- Congestion
    COALESCE(bf.concurrent_berth_calls, 0)                    AS concurrent_berth_calls,

    -- Weather
    wf.weather_severity,
    wf.wind_speed_knots,
    wf.weather_impact_factor,
    CAST(wf.storm_at_eta AS INTEGER)                          AS storm_at_eta,

    -- Temporal
    EXTRACT(month  FROM c.eta)                                AS eta_month,
    EXTRACT(isodow FROM c.eta)                                AS eta_dow,
    EXTRACT(hour   FROM c.eta)                                AS eta_hour,
    CAST((EXTRACT(isodow FROM c.eta) IN (6, 7)) AS INTEGER)   AS is_weekend,
    CAST((EXTRACT(hour FROM c.eta) >= 20
          OR EXTRACT(hour FROM c.eta) < 6) AS INTEGER)        AS is_night_arrival,

    -- Label (NULL for inference targets without actuals)
    CASE
        WHEN c.atd IS NULL THEN NULL            -- in-progress: inference target
        WHEN c.delay_hours > 0.5 THEN 1
        ELSE 0
    END                                                       AS is_delayed

FROM prediction_candidates c
LEFT JOIN vessel_feats   vf USING (vessel_call_id)
LEFT JOIN terminal_feats tf USING (vessel_call_id)
LEFT JOIN berth_feats    bf USING (vessel_call_id)
LEFT JOIN weather_feats  wf USING (vessel_call_id)
ORDER BY c.eta;
