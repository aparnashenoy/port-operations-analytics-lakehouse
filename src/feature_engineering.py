"""
Feature engineering pipeline for the vessel delay prediction model.

Reads Silver and Gold Parquet files and produces a single flat feature table
at vessel-call grain saved to data/gold/ml_vessel_delay_features.parquet.

Point-in-time (PIT) correctness
─────────────────────────────────
Every feature must represent information that would realistically be available
to an operations analyst at the moment the ETA notification is received — i.e.,
*before* the vessel physically arrives.  Using future actuals (ATA, ATD, actual
delay) as predictors produces a model that cannot be deployed.

Leakage budget per feature:
  • Static call attributes (terminal, carrier, vessel class, planned TEU,
    planned crane hours) — booked at scheduling time. ✓
  • Temporal features (day of week, month) — derived from ETA. ✓
  • Weather (storm_flag) — derived from the daily severity index for the ETA
    date, representing an observable forecast. ✓
  • Historical delay features — derived from completed calls whose ATD is
    strictly before the current call's ETA (no future leakage). ✓
  • Terminal congestion — derived from planned ETA/ETD of concurrent calls;
    actuals (ATA/ATD) are never used. ✓
  • Target (is_arrival_delayed_more_than_2_hours) — derived from
    arrival_delay_minutes, which is known only after arrival.  It is NULL
    for in-progress calls; those are inference targets, not training rows. ✓

Run:
    python src/feature_engineering.py
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    DELAY_THRESHOLD_HOURS,
    GOLD_DIR,
    LOG_LEVEL,
    SILVER_DIR,
    ensure_dirs,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of prior completed calls to average for rolling history features.
HISTORY_WINDOW = 10

# Severity index above this value classifies a day as a storm.
STORM_SEVERITY_THRESHOLD = 0.5

# Delay threshold for the binary label (hours → minutes).
LABEL_THRESHOLD_MINUTES = DELAY_THRESHOLD_HOURS * 60  # 120 minutes

# Ordered category lists for stable ordinal encoding.
# Explicit ordering ensures encoding values are consistent across train/inference.
TERMINAL_CATEGORIES = [["CPT", "EFT", "NCT", "SLH", "WIT"]]
SERVICE_CATEGORIES  = [["APL", "CMACGM", "COSCO", "EVERGREEN", "HAPAG",
                         "MAERSK", "MSC", "ONE", "PIL", "ZIM"]]

OUTPUT_PATH = GOLD_DIR / "ml_vessel_delay_features.parquet"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_inputs() -> dict[str, pd.DataFrame]:
    """Load all Silver and Gold tables required for feature computation."""
    paths = {
        "vessel_calls":     SILVER_DIR / "vessel_calls_silver.parquet",
        "crane_assignments":SILVER_DIR / "crane_assignments_silver.parquet",
        "weather_daily":    SILVER_DIR / "weather_daily_silver.parquet",
        "terminal_metadata":SILVER_DIR / "terminal_metadata_silver.parquet",
        "call_summary":     GOLD_DIR   / "gold_vessel_call_summary.parquet",
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}. "
                "Run silver_transformations.py and gold_kpis.py first."
            )
        tables[name] = pd.read_parquet(path, engine="pyarrow")
        log.debug("Loaded %-25s  %d rows", name, len(tables[name]))
    return tables


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def build_candidates(vc: pd.DataFrame) -> pd.DataFrame:
    """
    Select vessel calls that are valid prediction targets.

    Excluded:
      - Calls with missing ETA: no anchor for temporal features.
      - Calls with invalid arrival sequences: the target variable is corrupt.

    The returned frame contains only identifier and scheduling columns;
    no actual-outcome columns are carried forward here.
    """
    mask = (
        vc["eta"].notna()
        & ~vc["invalid_arrival_sequence_flag"]
    )
    candidates = vc.loc[mask, [
        "vessel_call_id",
        "vessel_id",
        "carrier_code",
        "terminal_id",
        "eta",
        "etd",
        "planned_cargo_teu",
        "planned_turnaround_hours",
        "arrival_delay_minutes",   # kept to derive target; NOT a feature
        "atd",                     # kept to derive target; NOT a feature
    ]].copy()

    log.info("Candidate calls: %d of %d (%.1f%%)",
             len(candidates), len(vc), 100 * len(candidates) / len(vc))
    return candidates.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature group: static call attributes
# ---------------------------------------------------------------------------


def add_planned_scope(
    df: pd.DataFrame,
    ca: pd.DataFrame,
    gs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach features derived from the vessel call booking and crane schedule.

    PIT: planned_cargo_teu is set at booking time.  planned_crane_count and
    planned_crane_hours come from planned_hours in crane_assignments (the
    pre-arrival booking), not from actual_hours (post-arrival actuals).
    planned_moves comes from gold_vessel_call_summary which aggregates
    container move records — the planned scope is determined before arrival.
    """
    df = df.copy()

    # vessel_capacity_teu: planned_cargo_teu is the closest available proxy
    # for vessel size (capacity × fill_rate; fill_rate ~ 0.55–0.98).
    df["vessel_capacity_teu"] = df["planned_cargo_teu"].astype(float)

    # --- Crane booking aggregates (use PLANNED hours, not actual) ---
    # PIT: crane assignments are bookings made when the vessel is scheduled.
    crane_agg = (
        ca.groupby("vessel_call_id")
        .agg(
            planned_crane_count=("crane_id",      "nunique"),
            planned_crane_hours=("planned_hours", "sum"),
        )
        .reset_index()
    )
    df = df.merge(crane_agg, on="vessel_call_id", how="left")

    # Calls with no crane bookings get 0 (vessel scheduled but no cranes yet assigned)
    df["planned_crane_count"] = df["planned_crane_count"].fillna(0).astype(int)
    df["planned_crane_hours"] = df["planned_crane_hours"].fillna(0.0)

    # --- Planned moves (scope known at booking time) ---
    # PIT: planned_moves represents the cargo plan, set before vessel arrival.
    moves = gs[["vessel_call_id", "planned_moves"]].copy()
    df = df.merge(moves, on="vessel_call_id", how="left")
    df["planned_moves"] = df["planned_moves"].fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# Feature group: temporal
# ---------------------------------------------------------------------------


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive calendar features from ETA.

    PIT: ETA is defined at scheduling time — no actuals involved.
    """
    df = df.copy()
    df["day_of_week"] = df["eta"].dt.dayofweek          # Mon=0, Sun=6
    df["month"]       = df["eta"].dt.month               # Jan=1, Dec=12
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    return df


# ---------------------------------------------------------------------------
# Feature group: weather
# ---------------------------------------------------------------------------


def add_weather_features(df: pd.DataFrame, wt: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the weather severity index for the ETA date at the vessel's terminal.

    PIT: weather observations for the ETA calendar day are available as a
    forecast at scheduling time.  No post-arrival weather is used.
    """
    df = df.copy()

    weather = wt[["terminal_id", "weather_date", "severity_index"]].copy()
    # Normalise to midnight so join works regardless of time-of-day stored in
    # weather_date (silver normalises it but guard for safety)
    weather["weather_date"] = pd.to_datetime(weather["weather_date"]).dt.normalize()
    weather["terminal_id"]  = weather["terminal_id"].astype(str)

    df["_eta_date"]   = df["eta"].dt.normalize()
    df["terminal_id"] = df["terminal_id"].astype(str)

    df = df.merge(
        weather.rename(columns={"weather_date": "_eta_date", "severity_index": "_severity"}),
        on=["terminal_id", "_eta_date"],
        how="left",
    )

    df["storm_flag"] = (df["_severity"] > STORM_SEVERITY_THRESHOLD).astype(int)
    df = df.drop(columns=["_eta_date", "_severity"])
    return df


# ---------------------------------------------------------------------------
# Feature group: historical delay rates (PIT self-join helpers)
# ---------------------------------------------------------------------------


def _pit_rolling_mean(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    group_col: str,
    value_col: str = "delay_hours",
    n: int = HISTORY_WINDOW,
) -> pd.Series:
    """
    For each candidate call, return the mean of value_col over the N most
    recent completed calls in history that share the same group_col value
    and whose ATD is strictly before the candidate's ETA.

    PIT guarantee enforced by:
        history["atd"] < candidates["eta"]

    This strict inequality means no call can "see" its own outcome or the
    outcome of any call that completed after it arrived.

    Returns a Series indexed by vessel_call_id.
    """
    # Minimal slices to keep the cross-merge small
    c = candidates[["vessel_call_id", group_col, "eta"]].copy()
    h = history[["vessel_call_id", group_col, "atd", value_col]].copy()
    h = h.rename(columns={
        "vessel_call_id": "_h_call_id",
        "atd": "_h_atd",
        value_col: "_h_value",
    })

    # Join on the grouping dimension (terminal or carrier)
    merged = c.merge(h, on=group_col, how="left")

    # PIT filter + self-exclusion
    # atd < eta ensures we never use a call's own completion time as context
    merged = merged[
        (merged["_h_atd"] < merged["eta"]) &
        (merged["_h_call_id"] != merged["vessel_call_id"])
    ]

    if merged.empty:
        return pd.Series(dtype=float, name=value_col)

    # Rank historical calls within each candidate by recency (most recent = 1)
    merged["_rank"] = (
        merged.groupby("vessel_call_id")["_h_atd"]
        .rank(ascending=False, method="first")
    )

    return (
        merged[merged["_rank"] <= n]
        .groupby("vessel_call_id")["_h_value"]
        .mean()
    )


def _pit_most_recent_value(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    group_col: str,
    value_col: str = "delay_hours",
) -> pd.Series:
    """
    Return the single most recent value of value_col in history for each
    candidate, constrained to records whose ATD is strictly before the
    candidate's ETA.

    PIT guarantee: same as _pit_rolling_mean.
    """
    c = candidates[["vessel_call_id", group_col, "eta"]].copy()
    h = history[["vessel_call_id", group_col, "atd", value_col]].copy()
    h = h.rename(columns={
        "vessel_call_id": "_h_call_id",
        "atd": "_h_atd",
        value_col: "_h_value",
    })

    merged = c.merge(h, on=group_col, how="left")
    merged = merged[
        (merged["_h_atd"] < merged["eta"]) &
        (merged["_h_call_id"] != merged["vessel_call_id"])
    ]

    if merged.empty:
        return pd.Series(dtype=float, name=value_col)

    # Keep only the single most recent record per candidate
    most_recent = (
        merged.sort_values("_h_atd", ascending=False)
        .drop_duplicates(subset="vessel_call_id", keep="first")
    )
    return most_recent.set_index("vessel_call_id")["_h_value"]


def add_historical_features(
    df: pd.DataFrame,
    vc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add three PIT-safe rolling history features.

    avg_previous_10_terminal_delays
        Mean delay_hours of the N most recent completed calls at the same
        terminal, with ATD < current ETA.  Captures whether this terminal
        is currently running behind or on schedule.

    avg_previous_10_service_delays
        Same computation grouped by carrier_code (shipping line / service).
        Captures whether this carrier's vessels are systematically late
        on their current rotation.

    previous_vessel_delay
        delay_hours from the single most recent completed call for the same
        vessel_id with ATD < current ETA.  A vessel that was just delayed
        is more likely to arrive late again (operational momentum).

    All three use the PIT cross-join pattern: history.atd < candidates.eta.
    Rows with no qualifying history receive NaN (impute in the model step).
    """
    df = df.copy()

    # History pool: completed calls with known ATD only.
    # Using delay_hours from vessel_calls_silver (raw delay attribution).
    # NOT using arrival_delay_minutes (that depends on ETA/ATA — post-arrival).
    history = vc.loc[
        vc["atd"].notna() & vc["status"].eq("completed"),
        ["vessel_call_id", "vessel_id", "carrier_code", "terminal_id", "atd", "delay_hours"]
    ].copy()

    log.info("History pool for rolling features: %d completed calls", len(history))

    log.info("Computing avg_previous_%d_terminal_delays ...", HISTORY_WINDOW)
    terminal_hist = _pit_rolling_mean(df, history, group_col="terminal_id")
    df["avg_previous_10_terminal_delays"] = df["vessel_call_id"].map(terminal_hist)

    log.info("Computing avg_previous_%d_service_delays ...", HISTORY_WINDOW)
    service_hist = _pit_rolling_mean(df, history, group_col="carrier_code")
    df["avg_previous_10_service_delays"] = df["vessel_call_id"].map(service_hist)

    log.info("Computing previous_vessel_delay ...")
    vessel_hist = _pit_most_recent_value(df, history, group_col="vessel_id")
    df["previous_vessel_delay"] = df["vessel_call_id"].map(vessel_hist)

    return df


# ---------------------------------------------------------------------------
# Feature group: terminal congestion
# ---------------------------------------------------------------------------


def add_congestion_score(
    df: pd.DataFrame,
    vc: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate how congested the terminal is when each vessel is due to arrive.

    terminal_congestion_score is the count of *other* vessel calls at the
    same terminal whose planned window (ETA … ETD) overlaps the current
    call's ETA.

    PIT guarantee: only planned times (ETA, ETD) are used — never ATA or ATD.
    This represents information that is available on a port schedule/manifest
    at the time of the ETA notification, before any actual arrival occurs.
    """
    df = df.copy()

    # All calls with valid planned windows
    all_calls = vc.loc[
        vc["eta"].notna() & vc["etd"].notna(),
        ["vessel_call_id", "terminal_id", "eta", "etd"]
    ].copy()

    c = df[["vessel_call_id", "terminal_id", "eta"]].copy()
    a = all_calls.rename(columns={
        "vessel_call_id": "_other_id",
        "terminal_id":    "_other_terminal",
        "eta":            "_other_eta",
        "etd":            "_other_etd",
    })

    # Join on terminal
    merged = c.merge(a, left_on="terminal_id", right_on="_other_terminal", how="left")

    # PIT filter: other call's planned window must include the current ETA.
    # Using ETD (planned departure) — not ATD — to avoid post-arrival leakage.
    congestion_mask = (
        (merged["_other_id"] != merged["vessel_call_id"]) &  # not self
        (merged["_other_eta"] <= merged["eta"]) &             # other arrived before us
        (merged["_other_etd"] >= merged["eta"])               # other hasn't left yet (planned)
    )
    congestion = (
        merged[congestion_mask]
        .groupby("vessel_call_id")["_other_id"]
        .count()
        .rename("terminal_congestion_score")
    )

    df["terminal_congestion_score"] = df["vessel_call_id"].map(congestion).fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Feature group: categorical encoding
# ---------------------------------------------------------------------------


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal-encode terminal_id and service_code (carrier_code).

    Category order is fixed via TERMINAL_CATEGORIES and SERVICE_CATEGORIES so
    the mapping is stable across separate training and inference runs.  The
    original string columns are kept alongside the encoded integer columns.

    service_code is derived from carrier_code (the shipping line is the
    operational proxy for the service/rotation in this dataset).
    """
    df = df.copy()
    df["service_code"] = df["carrier_code"].astype(str)

    # terminal_id
    t_enc = OrdinalEncoder(
        categories=TERMINAL_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    df["terminal_id_encoded"] = t_enc.fit_transform(
        df["terminal_id"].astype(str).to_numpy().reshape(-1, 1)
    ).astype(int)

    # service_code
    s_enc = OrdinalEncoder(
        categories=SERVICE_CATEGORIES,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    df["service_code_encoded"] = s_enc.fit_transform(
        df["service_code"].to_numpy().reshape(-1, 1)
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the binary classification label.

    is_arrival_delayed_more_than_2_hours = 1 when the vessel arrives more than
    120 minutes after its scheduled ETA.

    NULL for calls without a recorded ATA (in-progress or future calls); these
    rows are valid inference targets but must be excluded from model training.

    PIT: this column is the OUTCOME we are predicting.  It must never be used
    as a feature input.  It is included here only so training scripts can drop
    it before fitting and evaluate predictions after fitting.
    """
    df = df.copy()

    df["is_arrival_delayed_more_than_2_hours"] = np.where(
        df["arrival_delay_minutes"].isna(),
        np.nan,                                             # unknown (inference target)
        (df["arrival_delay_minutes"] > LABEL_THRESHOLD_MINUTES).astype(float),
    )

    return df


# ---------------------------------------------------------------------------
# Assembly and output
# ---------------------------------------------------------------------------


def assemble_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and order the final output columns.

    Columns are ordered: identifiers → categoricals → numeric call features
    → temporal → weather → historical → congestion → target.
    """
    feature_cols = [
        # Identifiers (drop before model training)
        "vessel_call_id",
        "eta",

        # Categorical features (raw string + encoded integer)
        "terminal_id",
        "terminal_id_encoded",
        "service_code",
        "service_code_encoded",

        # Static call features (known at booking time)
        "vessel_capacity_teu",
        "planned_moves",
        "planned_crane_count",
        "planned_crane_hours",

        # Temporal features (derived from ETA)
        "day_of_week",
        "month",
        "is_weekend",

        # Weather features (observable forecast at ETA date)
        "storm_flag",

        # Historical rolling features (PIT: history.atd < current.eta)
        "avg_previous_10_terminal_delays",
        "avg_previous_10_service_delays",
        "previous_vessel_delay",

        # Congestion (PIT: uses planned ETA/ETD only)
        "terminal_congestion_score",

        # Target (NULL for inference targets; drop before model training)
        "is_arrival_delayed_more_than_2_hours",
    ]
    return df[feature_cols].copy()


def write_features(df: pd.DataFrame) -> None:
    """Write the feature table to data/gold/ as Snappy-compressed Parquet."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, OUTPUT_PATH, compression="snappy")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame) -> None:
    total    = len(df)
    labeled  = df["is_arrival_delayed_more_than_2_hours"].notna().sum()
    delayed  = (df["is_arrival_delayed_more_than_2_hours"] == 1).sum()
    unlabeled = total - labeled

    print()
    print(f"  Feature table : {OUTPUT_PATH}")
    print(f"  Total rows    : {total:,}  (one per eligible vessel call)")
    print(f"  Labeled rows  : {labeled:,}  (completed calls — usable for training)")
    print(f"  Unlabeled rows: {unlabeled:,}  (in-progress calls — inference targets)")
    print(f"  Positive label: {delayed:,}  ({100*delayed/labeled:.1f}%  delayed > 2 h)")
    print(f"  Negative label: {int(labeled-delayed):,}  ({100*(labeled-delayed)/labeled:.1f}%  on-time / < 2 h delay)")
    print()
    print(f"  {'Feature':<42} {'Non-null':>8}  {'Mean':>8}  {'Std':>8}")
    print("  " + "-" * 72)
    numeric_cols = [
        "vessel_capacity_teu", "planned_moves", "planned_crane_count",
        "planned_crane_hours", "day_of_week", "month", "is_weekend",
        "storm_flag", "avg_previous_10_terminal_delays",
        "avg_previous_10_service_delays", "previous_vessel_delay",
        "terminal_congestion_score",
    ]
    for col in numeric_cols:
        s = df[col]
        nn = int(s.notna().sum())
        mean = f"{s.mean():.2f}" if nn > 0 else "—"
        std  = f"{s.std():.2f}"  if nn > 0 else "—"
        print(f"  {col:<42} {nn:>8,}  {mean:>8}  {std:>8}")

    print()
    print("  Encoding maps")
    print(f"  {'terminal_id':<15} CPT=0  EFT=1  NCT=2  SLH=3  WIT=4")
    print(f"  {'service_code':<15} APL=0  CMACGM=1  COSCO=2  EVERGREEN=3  HAPAG=4")
    print(f"                  MAERSK=5  MSC=6  ONE=7  PIL=8  ZIM=9")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_dirs()

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Feature engineering  |  {ts}", flush=True)

    log.info("Loading inputs ...")
    tables = load_inputs()
    vc  = tables["vessel_calls"]
    ca  = tables["crane_assignments"]
    wt  = tables["weather_daily"]
    gs  = tables["call_summary"]

    log.info("Building candidate call set ...")
    df = build_candidates(vc)

    log.info("Adding planned scope features ...")
    df = add_planned_scope(df, ca, gs)

    log.info("Adding temporal features ...")
    df = add_temporal_features(df)

    log.info("Adding weather features ...")
    df = add_weather_features(df, wt)

    log.info("Adding historical delay features ...")
    df = add_historical_features(df, vc)

    log.info("Adding terminal congestion score ...")
    df = add_congestion_score(df, vc)

    log.info("Encoding categoricals ...")
    df = encode_categoricals(df)

    log.info("Adding target variable ...")
    df = add_target(df)

    log.info("Assembling final feature table ...")
    df = assemble_feature_table(df)

    log.info("Writing to %s ...", OUTPUT_PATH)
    write_features(df)

    print_summary(df)


if __name__ == "__main__":
    main()
