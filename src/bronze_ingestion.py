"""
Bronze ingestion pipeline for the port operations analytics lakehouse.

Reads raw CSV files from data/raw/, applies lightweight schema enforcement,
stamps every record with ingestion metadata, and writes Parquet to
data/bronze/.

Bronze-layer philosophy:
  - Preserve source data faithfully; do not fix or drop data quality issues.
  - Enforce structural constraints only: expected columns present, dtypes
    castable, key columns non-null.
  - Route structurally broken rows (null primary keys, uncastable required
    fields) to data/quarantine/ so they are visible and auditable.
  - Leave business-level issues (duplicates, invalid sequences, missing
    optional fields) intact for the silver layer to handle.

Run:
    python src/bronze_ingestion.py
"""

import logging
import sys
import uuid
from dataclasses import dataclass
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
    QUARANTINE_DIR,
    RAW_DIR,
    ensure_dirs,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SOURCE_SYSTEM = "port_ops_raw"


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSpec:
    """
    Declares the expected shape of a single column at the bronze boundary.

    Attributes:
        name:     Column name as it appears in the source CSV.
        dtype:    Target pandas dtype string (e.g. "string", "float64",
                  "Int64", "datetime64[ns]").
        nullable: If False, a non-null invariant is expected; null rate is
                  logged as a warning but the row is not quarantined unless
                  is_key is also True.
        is_key:   If True, rows where this column is null are routed to
                  quarantine rather than bronze.
    """

    name: str
    dtype: str
    nullable: bool = True
    is_key: bool = False


# Ordered so foreign-key tables come after their parents.
SCHEMAS: dict[str, list[ColumnSpec]] = {
    "terminal_metadata": [
        ColumnSpec("terminal_id",            "string",        nullable=False, is_key=True),
        ColumnSpec("terminal_name",          "string",        nullable=False),
        ColumnSpec("berth_count",            "Int64",         nullable=False),
        ColumnSpec("crane_count",            "Int64",         nullable=False),
        ColumnSpec("max_vessel_class",       "string"),
        ColumnSpec("base_productivity_mph",  "float64"),
        ColumnSpec("delay_bias",             "float64"),
        ColumnSpec("latitude",               "float64"),
        ColumnSpec("longitude",              "float64"),
    ],
    "weather_daily": [
        ColumnSpec("weather_date",           "datetime64[ns]", nullable=False, is_key=True),
        ColumnSpec("terminal_id",            "string",         nullable=False, is_key=True),
        ColumnSpec("wind_speed_knots",       "float64"),
        ColumnSpec("wave_height_m",          "float64"),
        ColumnSpec("visibility_nm",          "float64"),
        ColumnSpec("precipitation_mm",       "float64"),
        ColumnSpec("severity_index",         "float64"),
        ColumnSpec("weather_impact_factor",  "float64"),
    ],
    "vessel_calls": [
        ColumnSpec("vessel_call_id",            "string",         nullable=False, is_key=True),
        ColumnSpec("vessel_id",                 "string",         nullable=False, is_key=True),
        ColumnSpec("vessel_name",               "string"),
        ColumnSpec("vessel_class",              "string"),
        ColumnSpec("carrier_code",              "string"),
        ColumnSpec("terminal_id",               "string",         nullable=False, is_key=True),
        ColumnSpec("berth_id",                  "string"),
        ColumnSpec("eta",                       "datetime64[ns]"),
        ColumnSpec("ata",                       "datetime64[ns]"),
        ColumnSpec("etd",                       "datetime64[ns]"),
        ColumnSpec("atd",                       "datetime64[ns]"),
        ColumnSpec("planned_turnaround_hours",  "float64"),
        ColumnSpec("actual_turnaround_hours",   "float64"),
        ColumnSpec("delay_hours",               "float64"),
        ColumnSpec("delay_reason",              "string"),
        ColumnSpec("planned_cargo_teu",         "Int64"),
        ColumnSpec("actual_cargo_teu",          "Int64"),
        ColumnSpec("weather_impact_factor",     "float64"),
        ColumnSpec("status",                    "string"),
        ColumnSpec("record_created_at",         "datetime64[ns]"),
    ],
    "crane_assignments": [
        ColumnSpec("assignment_id",               "string",         nullable=False, is_key=True),
        ColumnSpec("vessel_call_id",              "string",         nullable=False, is_key=True),
        ColumnSpec("terminal_id",                 "string",         nullable=False, is_key=True),
        ColumnSpec("crane_id",                    "string",         nullable=False),
        ColumnSpec("planned_start",               "datetime64[ns]"),
        ColumnSpec("planned_end",                 "datetime64[ns]"),
        ColumnSpec("actual_start",                "datetime64[ns]"),
        ColumnSpec("actual_end",                  "datetime64[ns]"),
        ColumnSpec("planned_hours",               "float64"),
        ColumnSpec("actual_hours",                "float64"),
        ColumnSpec("productivity_moves_per_hour", "float64"),
        ColumnSpec("status",                      "string"),
    ],
    "container_moves": [
        ColumnSpec("move_id",                   "string",         nullable=False, is_key=True),
        ColumnSpec("assignment_id",             "string",         nullable=False, is_key=True),
        ColumnSpec("vessel_call_id",            "string",         nullable=False, is_key=True),
        ColumnSpec("terminal_id",               "string",         nullable=False, is_key=True),
        ColumnSpec("crane_id",                  "string"),
        ColumnSpec("move_type",                 "string"),
        ColumnSpec("container_size",            "string"),
        ColumnSpec("container_type",            "string"),
        ColumnSpec("planned_move_time",         "datetime64[ns]"),
        ColumnSpec("actual_move_time",          "datetime64[ns]"),
        ColumnSpec("planned_duration_minutes",  "float64"),
        ColumnSpec("actual_duration_minutes",   "float64"),
        ColumnSpec("move_status",               "string"),
    ],
}

INGEST_ORDER = [
    "terminal_metadata",
    "weather_daily",
    "vessel_calls",
    "crane_assignments",
    "container_moves",
]


# ---------------------------------------------------------------------------
# Column-level casting
# ---------------------------------------------------------------------------


def _cast_column(series: pd.Series, spec: ColumnSpec) -> tuple[pd.Series, int]:
    """
    Cast a Series to the dtype declared in spec, coercing unparseable values
    to null rather than raising.

    Returns:
        (cast_series, n_coerced) — n_coerced is the count of values that
        could not be cast and were replaced with null.
    """
    original_null_count = int(series.isna().sum())

    try:
        if spec.dtype == "datetime64[ns]":
            cast = pd.to_datetime(series, errors="coerce")
        elif spec.dtype in ("Int64", "int64"):
            cast = pd.to_numeric(series, errors="coerce").astype("Int64")
        elif spec.dtype == "float64":
            cast = pd.to_numeric(series, errors="coerce").astype("float64")
        elif spec.dtype == "string":
            cast = series.where(series.isna(), series.astype(str)).astype("string")
        else:
            cast = series.astype(spec.dtype)
    except Exception as exc:  # noqa: BLE001
        log.warning("    Could not cast column '%s' to %s: %s", spec.name, spec.dtype, exc)
        return series.copy(), 0

    new_null_count = int(cast.isna().sum())
    n_coerced = max(0, new_null_count - original_null_count)
    return cast, n_coerced


# ---------------------------------------------------------------------------
# Row-level quarantine routing
# ---------------------------------------------------------------------------


def _split_on_key_nulls(
    df: pd.DataFrame,
    specs: list[ColumnSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate rows where any is_key column contains a null value.

    Returns:
        (accepted_df, quarantine_df) — both are independent copies.
    """
    key_cols = [s.name for s in specs if s.is_key and s.name in df.columns]
    if not key_cols:
        return df.copy(), pd.DataFrame(columns=df.columns)

    bad_mask = df[key_cols].isna().any(axis=1)
    return df[~bad_mask].copy(), df[bad_mask].copy()


# ---------------------------------------------------------------------------
# Metadata stamping
# ---------------------------------------------------------------------------


def _add_metadata(
    df: pd.DataFrame,
    *,
    batch_id: str,
    ingestion_ts: datetime,
    source_file: str,
) -> pd.DataFrame:
    """
    Append four ingestion metadata columns to a DataFrame in-place.
    Modifies and returns the same object.
    """
    df["_ingestion_ts"] = pd.Timestamp(ingestion_ts)
    df["_source_system"] = SOURCE_SYSTEM
    df["_batch_id"] = batch_id
    df["_source_file"] = source_file
    return df


# ---------------------------------------------------------------------------
# Parquet writers
# ---------------------------------------------------------------------------


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to Parquet using Snappy compression."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


# ---------------------------------------------------------------------------
# Per-table ingestion
# ---------------------------------------------------------------------------


def ingest_table(
    table_name: str,
    batch_id: str,
    ingestion_ts: datetime,
) -> dict[str, Any]:
    """
    Ingest one raw CSV into the bronze layer.

    Steps:
      1. Read the source CSV.
      2. Add any declared columns that are absent (filled with null).
      3. Cast each column to its declared dtype, coercing bad values to null.
      4. Warn on unexpected null rates for non-nullable columns.
      5. Route rows with null key columns to quarantine.
      6. Stamp accepted and quarantine rows with ingestion metadata.
      7. Write accepted rows to data/bronze/<table_name>.parquet.
      8. Write quarantined rows to data/quarantine/<table_name>_<batch>.parquet.

    Returns:
        Summary dict with counts for the final report.
    """
    csv_path = RAW_DIR / f"{table_name}.csv"
    if not csv_path.exists():
        log.warning("Source file not found, skipping: %s", csv_path)
        return {
            "table": table_name,
            "status": "skipped",
            "rows_raw": 0,
            "rows_accepted": 0,
            "rows_quarantined": 0,
            "values_coerced": 0,
            "missing_cols": [],
        }

    specs = SCHEMAS[table_name]
    log.info("Ingesting  %-22s  ←  %s", table_name, csv_path.name)

    # 1. Read raw CSV
    df = pd.read_csv(csv_path, low_memory=False)
    rows_raw = len(df)

    # 2. Reconcile columns
    expected_cols = {s.name for s in specs}
    actual_cols = set(df.columns)

    missing_cols = sorted(expected_cols - actual_cols)
    if missing_cols:
        log.warning("    Missing expected columns (added as null): %s", missing_cols)
        for col in missing_cols:
            df[col] = None

    extra_cols = sorted(actual_cols - expected_cols)
    if extra_cols:
        log.debug("    Extra columns carried through: %s", extra_cols)

    # 3. Cast each column to its declared dtype
    total_coerced = 0
    for spec in specs:
        df[spec.name], n_coerced = _cast_column(df[spec.name], spec)
        if n_coerced > 0:
            log.debug("    %-35s coerced %d values to null", spec.name, n_coerced)
        total_coerced += n_coerced

    # 4. Warn on unexpected nulls in non-nullable columns
    for spec in specs:
        if not spec.nullable:
            null_count = int(df[spec.name].isna().sum())
            if null_count > 0:
                log.warning(
                    "    Non-nullable column '%s' has %d null(s) (%.1f%%)",
                    spec.name,
                    null_count,
                    100 * null_count / max(1, rows_raw),
                )

    # 5. Route null-key rows to quarantine
    df_accepted, df_quarantine = _split_on_key_nulls(df, specs)

    # 6. Stamp metadata on both partitions
    meta_kwargs = dict(batch_id=batch_id, ingestion_ts=ingestion_ts, source_file=csv_path.name)
    _add_metadata(df_accepted, **meta_kwargs)
    if not df_quarantine.empty:
        _add_metadata(df_quarantine, **meta_kwargs)
        df_quarantine["_quarantine_reason"] = "null_key_column"

    # 7. Write accepted rows to bronze
    bronze_path = BRONZE_DIR / f"{table_name}.parquet"
    if not df_accepted.empty:
        _write_parquet(df_accepted, bronze_path)
        log.debug("    Wrote %d rows  →  %s", len(df_accepted), bronze_path)

    # 8. Write quarantined rows
    rows_quarantined = len(df_quarantine)
    if rows_quarantined > 0:
        q_path = QUARANTINE_DIR / f"{table_name}_{batch_id[:8]}.parquet"
        _write_parquet(df_quarantine, q_path)
        log.warning(
            "    Quarantined %d rows  →  %s", rows_quarantined, q_path.name
        )

    return {
        "table":            table_name,
        "status":           "ok",
        "rows_raw":         rows_raw,
        "rows_accepted":    len(df_accepted),
        "rows_quarantined": rows_quarantined,
        "values_coerced":   total_coerced,
        "missing_cols":     missing_cols,
    }


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict[str, Any]], batch_id: str) -> None:
    """Print a formatted ingestion summary to stdout."""
    col_w = 22
    print()
    print(f"  {'Table':<{col_w}} {'Raw':>8}  {'Accepted':>9}  {'Quarantined':>12}  {'Coerced':>8}")
    print("  " + "-" * 68)

    totals: dict[str, int] = {"rows_raw": 0, "rows_accepted": 0, "rows_quarantined": 0, "values_coerced": 0}

    for r in results:
        if r["status"] == "skipped":
            print(f"  {r['table']:<{col_w}} {'—':>8}  {'skipped':>9}")
            continue
        print(
            f"  {r['table']:<{col_w}} {r['rows_raw']:>8,}  {r['rows_accepted']:>9,}  "
            f"{r['rows_quarantined']:>12,}  {r['values_coerced']:>8,}"
        )
        for key in totals:
            totals[key] += r.get(key, 0)

    print("  " + "-" * 68)
    print(
        f"  {'TOTAL':<{col_w}} {totals['rows_raw']:>8,}  {totals['rows_accepted']:>9,}  "
        f"{totals['rows_quarantined']:>12,}  {totals['values_coerced']:>8,}"
    )
    print()
    print(f"  Batch ID  : {batch_id}")
    print(f"  Bronze    : {BRONZE_DIR}")
    if totals["rows_quarantined"] > 0:
        print(f"  Quarantine: {QUARANTINE_DIR}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_dirs()

    batch_id = str(uuid.uuid4())
    ingestion_ts = datetime.now(tz=timezone.utc)

    print(
        f"Bronze ingestion  |  batch {batch_id[:8]}  |  "
        f"{ingestion_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for table_name in INGEST_ORDER:
        result = ingest_table(table_name, batch_id, ingestion_ts)
        results.append(result)

    _print_summary(results, batch_id)


if __name__ == "__main__":
    main()
