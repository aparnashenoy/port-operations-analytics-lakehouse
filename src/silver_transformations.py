"""
Silver transformation pipeline for the port operations analytics lakehouse.

Reads bronze Parquet files from data/bronze/, applies business-level cleaning
and enrichment, and writes Silver Parquet files to data/silver/.

Silver-layer philosophy:
  - Deduplicate on business keys using latest-record-wins semantics.
  - Standardise timestamps and compute derived metrics (delay minutes,
    crane hours, duration variance).
  - Attach per-row boolean quality flags for every known issue category.
  - Never silently drop flagged rows; they remain in silver so analysts can
    filter, count, and investigate them directly.
  - Pass-through tables (terminal_metadata, weather_daily) receive metadata
    stamps and minor cleanup only.

Run:
    python src/silver_transformations.py
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    BRONZE_DIR,
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

# Delay in hours beyond which a vessel call is flagged as an outlier.
# Computed dynamically per batch (IQR method) but floored at this value.
OUTLIER_DELAY_FLOOR_HOURS = 24.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_bronze(table_name: str) -> pd.DataFrame:
    """Read a bronze Parquet file and return a DataFrame."""
    path = BRONZE_DIR / f"{table_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Bronze file not found: {path}")
    return pd.read_parquet(path, engine="pyarrow")


def write_silver(df: pd.DataFrame, table_name: str) -> Path:
    """Write a DataFrame to data/silver/ as a Snappy-compressed Parquet file."""
    path = SILVER_DIR / f"{table_name}.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    return path


def add_silver_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp a _silver_ts column on every row to record when the transform ran."""
    df = df.copy()
    df["_silver_ts"] = pd.Timestamp(datetime.now(tz=timezone.utc))
    return df


# ---------------------------------------------------------------------------
# vessel_calls transforms
# ---------------------------------------------------------------------------


def deduplicate_vessel_calls(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Retain the most recently updated record for each vessel_call_id.

    Sort priority: record_created_at DESC → _ingestion_ts DESC.
    Returns (deduped_df, n_dropped).
    """
    # Make _ingestion_ts timezone-naive for a uniform sort key
    df = df.copy()
    df["_ingestion_ts_utc"] = df["_ingestion_ts"].dt.tz_convert(None)

    df_sorted = df.sort_values(
        by=["vessel_call_id", "record_created_at", "_ingestion_ts_utc"],
        ascending=[True, False, False],
        na_position="last",
    )
    df_deduped = df_sorted.drop_duplicates(subset=["vessel_call_id"], keep="first")
    df_deduped = df_deduped.drop(columns=["_ingestion_ts_utc"])

    n_dropped = len(df) - len(df_deduped)
    return df_deduped.reset_index(drop=True), n_dropped


def compute_delay_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add arrival_delay_minutes and departure_delay_minutes columns.

    - arrival_delay_minutes  : (ata - eta) in minutes. Positive = late arrival.
    - departure_delay_minutes: (atd - etd) in minutes. Positive = late departure.

    Both are null when either operand timestamp is missing.
    """
    df = df.copy()

    df["arrival_delay_minutes"] = (
        (df["ata"] - df["eta"]).dt.total_seconds() / 60
    ).where(df["ata"].notna() & df["eta"].notna())

    df["departure_delay_minutes"] = (
        (df["atd"] - df["etd"]).dt.total_seconds() / 60
    ).where(df["atd"].notna() & df["etd"].notna())

    return df


def _outlier_threshold(series: pd.Series) -> float:
    """
    Compute the Tukey upper fence on the distribution of positive delay values.
    Returns the larger of the computed fence and OUTLIER_DELAY_FLOOR_HOURS.
    """
    positive = series[series > 0].dropna()
    if positive.empty:
        return OUTLIER_DELAY_FLOOR_HOURS
    q1, q3 = positive.quantile(0.25), positive.quantile(0.75)
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    return max(fence, OUTLIER_DELAY_FLOOR_HOURS)


def add_vessel_call_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach boolean data quality flag columns to vessel_calls.

    Flags:
        missing_eta_flag               — eta is null.
        missing_ata_flag               — ata is null.
        invalid_arrival_sequence_flag  — ata > atd (arrived after departing).
        invalid_departure_sequence_flag — etd < eta (planned departure before
                                          planned arrival; data entry error).
        large_delay_outlier_flag       — delay_hours exceeds the Tukey upper
                                          fence of observed positive delays
                                          (floor: OUTLIER_DELAY_FLOOR_HOURS h).
    """
    df = df.copy()

    df["missing_eta_flag"] = df["eta"].isna()
    df["missing_ata_flag"] = df["ata"].isna()

    df["invalid_arrival_sequence_flag"] = (
        df["ata"].notna() & df["atd"].notna() & (df["ata"] > df["atd"])
    )

    df["invalid_departure_sequence_flag"] = (
        df["eta"].notna() & df["etd"].notna() & (df["etd"] < df["eta"])
    )

    threshold = _outlier_threshold(df["delay_hours"])
    log.debug("    delay outlier threshold: %.1f hours", threshold)
    df["large_delay_outlier_flag"] = df["delay_hours"].notna() & (
        df["delay_hours"] > threshold
    )

    return df


def transform_vessel_calls(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply all silver transforms to vessel_calls.

    Returns (transformed_df, stats_dict) where stats_dict contains row counts
    and flag summaries used in the final report.
    """
    rows_in = len(df)

    df, n_dupes_dropped = deduplicate_vessel_calls(df)
    df = compute_delay_minutes(df)
    df = add_vessel_call_flags(df)
    df = add_silver_metadata(df)

    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    flag_counts = {col: int(df[col].sum()) for col in flag_cols}

    return df, {
        "rows_in":       rows_in,
        "rows_out":      len(df),
        "dupes_dropped": n_dupes_dropped,
        "flag_counts":   flag_counts,
    }


# ---------------------------------------------------------------------------
# crane_assignments transforms
# ---------------------------------------------------------------------------


def compute_crane_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add planned_crane_hours and actual_crane_hours derived from the explicit
    timestamp columns.  These supersede the pre-computed float columns from
    bronze, which may reflect planning figures rather than true durations.
    """
    df = df.copy()

    df["planned_crane_hours"] = (
        (df["planned_end"] - df["planned_start"]).dt.total_seconds() / 3600
    ).where(df["planned_start"].notna() & df["planned_end"].notna())

    df["actual_crane_hours"] = (
        (df["actual_end"] - df["actual_start"]).dt.total_seconds() / 3600
    ).where(df["actual_start"].notna() & df["actual_end"].notna())

    return df


def detect_crane_overlaps(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean Series where True marks assignments that participate in
    a temporal overlap with another assignment for the same crane.

    Algorithm:
      1. Filter to rows with valid, non-inverted actual_start/actual_end.
      2. Sort by (crane_id, actual_start).
      3. Within each crane group, compare each row's actual_start against the
         cumulative maximum actual_end of all preceding rows.  Any row that
         starts before that high-water mark overlaps a prior assignment.
      4. Back-propagate to flag the preceding assignment as well (using a
         next-row actual_start check via shift(-1)).
    """
    valid = (
        df["actual_start"].notna()
        & df["actual_end"].notna()
        & (df["actual_end"] >= df["actual_start"])
    )
    df_check = df.loc[valid, ["crane_id", "actual_start", "actual_end"]].copy()
    df_check = df_check.sort_values(["crane_id", "actual_start"])

    # No valid assignments remain after filtering — no overlaps possible.
    if df_check.empty:
        return pd.Series(False, index=df.index)

    # Running max of actual_end seen so far for each crane, shifted so the
    # current row compares against all *prior* assignments, not itself.
    df_check["_prior_max_end"] = df_check.groupby("crane_id")["actual_end"].transform(
        lambda s: s.cummax().shift(1)
    )
    # Next assignment's start time within the same crane (to flag earlier party)
    df_check["_next_start"] = df_check.groupby("crane_id")["actual_start"].shift(-1)

    overlaps_as_later = df_check["actual_start"] < df_check["_prior_max_end"]
    overlaps_as_earlier = df_check["actual_end"] > df_check["_next_start"]

    overlap_idx = set(df_check[overlaps_as_later | overlaps_as_earlier].index)
    return df.index.isin(overlap_idx)


def add_crane_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach boolean data quality flag columns to crane_assignments.

    Flags:
        missing_crane_id_flag   — crane_id is null.
        invalid_crane_time_flag — actual_start is null, actual_end is null,
                                  or actual_end < actual_start.
        crane_overlap_flag      — this assignment overlaps another assignment
                                  for the same crane.
    """
    df = df.copy()

    df["missing_crane_id_flag"] = df["crane_id"].isna()

    df["invalid_crane_time_flag"] = (
        df["actual_start"].isna()
        | df["actual_end"].isna()
        | (df["actual_end"] < df["actual_start"])
    )

    df["crane_overlap_flag"] = detect_crane_overlaps(df)

    return df


def transform_crane_assignments(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply all silver transforms to crane_assignments."""
    rows_in = len(df)

    df = compute_crane_hours(df)
    df = add_crane_flags(df)
    df = add_silver_metadata(df)

    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    flag_counts = {col: int(df[col].sum()) for col in flag_cols}

    return df, {
        "rows_in":     rows_in,
        "rows_out":    len(df),
        "flag_counts": flag_counts,
    }


# ---------------------------------------------------------------------------
# container_moves transforms
# ---------------------------------------------------------------------------


def clean_container_moves(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix structural issues in container_moves:

    - Null out negative actual_duration_minutes (data entry error; the silver
      value is null rather than an impossible negative, preserving the planned
      figure for imputation by downstream consumers).
    - Add duration_variance_minutes = actual - planned (null when either is
      missing or when actual was nulled out above).
    """
    df = df.copy()

    negative_mask = df["actual_duration_minutes"].notna() & (
        df["actual_duration_minutes"] < 0
    )
    df.loc[negative_mask, "actual_duration_minutes"] = pd.NA

    df["duration_variance_minutes"] = (
        (df["actual_duration_minutes"] - df["planned_duration_minutes"])
        .where(df["actual_duration_minutes"].notna() & df["planned_duration_minutes"].notna())
    )

    return df, int(negative_mask.sum())


def transform_container_moves(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply all silver transforms to container_moves."""
    rows_in = len(df)

    df, n_negative_nulled = clean_container_moves(df)
    df = add_silver_metadata(df)

    return df, {
        "rows_in":         rows_in,
        "rows_out":        len(df),
        "negative_nulled": n_negative_nulled,
        "flag_counts":     {},
    }


# ---------------------------------------------------------------------------
# Pass-through transforms
# ---------------------------------------------------------------------------


def transform_terminal_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Terminal metadata is a static reference table.  Silver applies metadata
    stamping only; no rows are modified.
    """
    df = add_silver_metadata(df)
    return df, {"rows_in": len(df), "rows_out": len(df), "flag_counts": {}}


def transform_weather_daily(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Weather data is append-only source truth.  Silver normalises weather_date
    to a date string (YYYY-MM-DD) for DuckDB compatibility and stamps metadata.
    """
    df = df.copy()
    df["weather_date"] = pd.to_datetime(df["weather_date"]).dt.normalize()
    df = add_silver_metadata(df)
    return df, {"rows_in": len(df), "rows_out": len(df), "flag_counts": {}}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

PIPELINE: list[tuple[str, str, Any]] = [
    ("terminal_metadata",  "terminal_metadata_silver",  transform_terminal_metadata),
    ("weather_daily",      "weather_daily_silver",      transform_weather_daily),
    ("vessel_calls",       "vessel_calls_silver",       transform_vessel_calls),
    ("crane_assignments",  "crane_assignments_silver",  transform_crane_assignments),
    ("container_moves",    "container_moves_silver",    transform_container_moves),
]


def _print_summary(report: list[dict[str, Any]]) -> None:
    print()
    print(f"  {'Table':<30} {'In':>7}  {'Out':>7}  {'Δ':>5}  Notes")
    print("  " + "-" * 80)

    for r in report:
        delta = r["rows_out"] - r["rows_in"]
        delta_str = f"{delta:+d}" if delta != 0 else "  —"
        notes: list[str] = []

        if r.get("dupes_dropped", 0):
            notes.append(f"{r['dupes_dropped']} dupes dropped")
        if r.get("negative_nulled", 0):
            notes.append(f"{r['negative_nulled']} negative durations nulled")
        for flag, count in r.get("flag_counts", {}).items():
            if count:
                short = flag.replace("_flag", "").replace("_", " ")
                notes.append(f"{count} {short}")

        note_str = " | ".join(notes) if notes else ""
        print(
            f"  {r['table']:<30} {r['rows_in']:>7,}  {r['rows_out']:>7,}  "
            f"{delta_str:>5}  {note_str}"
        )

    print()
    print(f"  Silver Parquet files written to: {SILVER_DIR}")


def main() -> None:
    ensure_dirs()

    silver_ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Silver transformations  |  {silver_ts}", flush=True)

    report: list[dict[str, Any]] = []

    for bronze_name, silver_name, transform_fn in PIPELINE:
        log.info("Transforming  %-28s  →  %s", bronze_name, silver_name)
        df_bronze = read_bronze(bronze_name)
        df_silver, stats = transform_fn(df_bronze)
        write_silver(df_silver, silver_name)
        stats["table"] = silver_name
        report.append(stats)
        log.debug("    %d rows in  →  %d rows out", stats["rows_in"], stats["rows_out"])

    _print_summary(report)


if __name__ == "__main__":
    main()
