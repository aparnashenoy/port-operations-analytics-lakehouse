"""
Gold KPI pipeline for the port operations analytics lakehouse.

Reads clean Silver Parquet files from data/silver/, joins and aggregates
them into two business-ready Gold tables, and writes Parquet to data/gold/.
A sample CSV is also written to outputs/ for quick inspection.

Gold tables produced:
  gold_daily_terminal_kpis.parquet
      Grain: one row per (terminal_id, operation_date).
      Covers vessel volume, delay performance, container throughput,
      crane utilisation, data quality counts, and storm exposure.

  gold_vessel_call_summary.parquet
      Grain: one row per vessel call.
      Enriches each call with terminal name, aggregated crane and move
      metrics, storm flag, and a delay category label.

Enrichment strategy:
  - Crane and move data are pre-aggregated to vessel_call_id level before
    any join, keeping the working DataFrame at ~2 k rows throughout.
  - Weather is joined on (terminal_id, operation_date) after deriving the
    operation date from ATA (fallback: ETA) on each vessel call.
  - All joins are left joins; calls with no crane or move records receive
    null metrics rather than being silently dropped.

Run:
    python src/gold_kpis.py
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    GOLD_DIR,
    LOG_LEVEL,
    PROJECT_ROOT,
    SILVER_DIR,
    ensure_dirs,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# A vessel call is "delayed" when arrival_delay_minutes exceeds this value.
DELAY_THRESHOLD_MINUTES = 30.0

# Weather severity_index above this value classifies a terminal-day as a storm.
STORM_SEVERITY_THRESHOLD = 0.5

# Ordered bin edges and labels for delay_category on vessel calls.
DELAY_BINS = [-np.inf, -30.0, 30.0, 120.0, 360.0, np.inf]
DELAY_LABELS = ["early", "on_time", "minor_delay", "moderate_delay", "major_delay"]

# Quality flag columns present on each silver table.
VC_FLAG_COLS = [
    "missing_eta_flag",
    "missing_ata_flag",
    "invalid_arrival_sequence_flag",
    "invalid_departure_sequence_flag",
    "large_delay_outlier_flag",
]
CA_FLAG_COLS = [
    "missing_crane_id_flag",
    "invalid_crane_time_flag",
    "crane_overlap_flag",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_silver() -> dict[str, pd.DataFrame]:
    """Load all required Silver Parquet files and return them as a dict."""
    names = [
        "vessel_calls_silver",
        "crane_assignments_silver",
        "container_moves_silver",
        "terminal_metadata_silver",
        "weather_daily_silver",
    ]
    tables: dict[str, pd.DataFrame] = {}
    for name in names:
        path = SILVER_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Silver file not found: {path}. "
                "Run src/silver_transformations.py first."
            )
        tables[name] = pd.read_parquet(path, engine="pyarrow")
        log.debug("Loaded %-30s  %d rows", name, len(tables[name]))
    return tables


def write_gold(df: pd.DataFrame, table_name: str) -> Path:
    """Write a Gold DataFrame to data/gold/ as Snappy-compressed Parquet."""
    path = GOLD_DIR / f"{table_name}.parquet"
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        path,
        compression="snappy",
    )
    return path


def add_gold_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp a _gold_ts column so consumers know when the layer was built."""
    df = df.copy()
    df["_gold_ts"] = pd.Timestamp(datetime.now(tz=timezone.utc))
    return df


# ---------------------------------------------------------------------------
# Pre-aggregation: crane metrics per vessel call
# ---------------------------------------------------------------------------


def agg_crane_per_call(ca: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate crane_assignments_silver to vessel_call_id level.

    Returns one row per vessel call with:
        total_crane_hours       — sum of all actual_crane_hours.
        valid_crane_hours       — sum of actual_crane_hours excluding
                                  assignments flagged as invalid_crane_time.
        cranes_deployed         — count of distinct crane IDs assigned.
        crane_dq_issue_count    — total data quality flag hits across all
                                  assignments for the call.
    """
    ca = ca.copy()

    # Compute row-level DQ score before aggregation
    ca["_crane_dq_row"] = ca[CA_FLAG_COLS].sum(axis=1)

    # Clamp negative actual_crane_hours to zero (caused by inverted timestamps
    # flagged as invalid_crane_time) so they do not deflate totals.
    ca["_safe_crane_hours"] = ca["actual_crane_hours"].clip(lower=0.0)

    # valid_crane_hours further excludes assignments flagged invalid entirely
    ca["_valid_crane_hours"] = ca["_safe_crane_hours"].where(
        ~ca["invalid_crane_time_flag"], other=0.0
    )

    agg = (
        ca.groupby("vessel_call_id", as_index=False)
        .agg(
            total_crane_hours   =("_safe_crane_hours",   "sum"),
            valid_crane_hours   =("_valid_crane_hours",  "sum"),
            cranes_deployed     =("crane_id",            "nunique"),
            crane_dq_issue_count=("_crane_dq_row",       "sum"),
        )
    )
    return agg


# ---------------------------------------------------------------------------
# Pre-aggregation: container move metrics per vessel call
# ---------------------------------------------------------------------------


def agg_moves_per_call(mv: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate container_moves_silver to vessel_call_id level.

    Returns one row per vessel call with:
        planned_moves   — total rows (planned scope, all statuses).
        actual_moves    — completed moves only.
    """
    planned = mv.groupby("vessel_call_id")["move_id"].count().rename("planned_moves")
    actual = (
        mv[mv["move_status"] == "completed"]
        .groupby("vessel_call_id")["move_id"]
        .count()
        .rename("actual_moves")
    )
    return pd.concat([planned, actual], axis=1).reset_index().fillna({"actual_moves": 0})


# ---------------------------------------------------------------------------
# Core enrichment: join vessel calls with all dimensions
# ---------------------------------------------------------------------------


def enrich_vessel_calls(
    vc: pd.DataFrame,
    ca_agg: pd.DataFrame,
    mv_agg: pd.DataFrame,
    wt: pd.DataFrame,
    tm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one enriched row per vessel call by joining crane aggregates,
    move aggregates, weather, and terminal metadata.

    Derived columns added here:
        operation_date   — ATA date (fallback: ETA date) for day-level joins.
        service_code     — carrier_code used as the shipping-service proxy.
        storm_flag       — True when weather severity_index > STORM_SEVERITY_THRESHOLD.
        delay_category   — binned label derived from arrival_delay_minutes.
        moves_per_crane_hour — actual_moves / valid_crane_hours (null-safe).
        vc_dq_issue_count    — count of quality flags set on this vessel call.
    """
    df = vc.copy()

    # --- operation_date: prefer ATA, fall back to ETA ---
    df["operation_date"] = pd.to_datetime(
        df["ata"].where(df["ata"].notna(), df["eta"])
    ).dt.normalize()  # midnight of the day, keeps it as datetime64 for Parquet

    # --- service_code: carrier as service proxy ---
    df["service_code"] = df["carrier_code"]

    # --- DQ issue count from vessel_calls flags ---
    df["vc_dq_issue_count"] = df[VC_FLAG_COLS].sum(axis=1)

    # --- Joins ---
    # Crane aggregates (left join: calls with no cranes get nulls)
    df = df.merge(ca_agg, on="vessel_call_id", how="left")

    # Move aggregates (left join)
    df = df.merge(mv_agg, on="vessel_call_id", how="left")

    # Terminal metadata (left join on terminal_id)
    tm_slim = tm[["terminal_id", "terminal_name", "crane_count"]].copy()
    tm_slim["terminal_id"] = tm_slim["terminal_id"].astype(str)
    df["terminal_id"] = df["terminal_id"].astype(str)
    df = df.merge(tm_slim, on="terminal_id", how="left")

    # Weather: join on terminal_id + operation_date
    wt_slim = wt[["terminal_id", "weather_date", "severity_index", "weather_impact_factor"]].copy()
    wt_slim["terminal_id"] = wt_slim["terminal_id"].astype(str)
    wt_slim["weather_date"] = pd.to_datetime(wt_slim["weather_date"]).dt.normalize()
    df = df.merge(
        wt_slim,
        left_on=["terminal_id", "operation_date"],
        right_on=["terminal_id", "weather_date"],
        how="left",
    ).drop(columns=["weather_date"], errors="ignore")

    # --- Derived metrics ---
    df["storm_flag"] = df["severity_index"].gt(STORM_SEVERITY_THRESHOLD)

    df["moves_per_crane_hour"] = (
        df["actual_moves"] / df["valid_crane_hours"].replace(0, np.nan)
    ).round(2)

    df["delay_category"] = pd.cut(
        df["arrival_delay_minutes"],
        bins=DELAY_BINS,
        labels=DELAY_LABELS,
        right=True,
    ).astype("string")

    # Combined DQ issue count (vessel call flags + crane flags)
    df["total_dq_issue_count"] = (
        df["vc_dq_issue_count"] + df["crane_dq_issue_count"].fillna(0)
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Gold table 1: daily terminal KPIs
# ---------------------------------------------------------------------------


def build_daily_terminal_kpis(enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the enriched vessel-call frame to (terminal_id, operation_date)
    grain and compute all requested KPI columns.

    Columns produced match the spec exactly:
        operation_date, terminal_id, terminal_name,
        vessel_calls, avg_arrival_delay_minutes, avg_departure_delay_minutes,
        delayed_vessel_percentage, total_container_moves, total_crane_hours,
        avg_moves_per_crane_hour, crane_utilization_hours,
        data_quality_issue_count, storm_days_count.
    """
    # Drop rows with no operation_date (both ATA and ETA null)
    df = enriched.dropna(subset=["operation_date"]).copy()

    # delayed_vessel flag: arrived more than DELAY_THRESHOLD_MINUTES late
    df["_is_delayed"] = (
        df["arrival_delay_minutes"].notna()
        & df["arrival_delay_minutes"].gt(DELAY_THRESHOLD_MINUTES)
    )

    kpis = (
        df.groupby(["terminal_id", "operation_date"], as_index=False)
        .agg(
            terminal_name              =("terminal_name",             "first"),
            vessel_calls               =("vessel_call_id",            "count"),
            avg_arrival_delay_minutes  =("arrival_delay_minutes",     "mean"),
            avg_departure_delay_minutes=("departure_delay_minutes",   "mean"),
            _delayed_count             =("_is_delayed",               "sum"),
            total_container_moves      =("actual_moves",              "sum"),
            total_crane_hours          =("total_crane_hours",         "sum"),
            crane_utilization_hours    =("valid_crane_hours",         "sum"),
            data_quality_issue_count   =("total_dq_issue_count",      "sum"),
            _storm_flag_any            =("storm_flag",                "max"),
        )
    )

    # delayed_vessel_percentage: % of calls in the group that were delayed
    kpis["delayed_vessel_percentage"] = (
        kpis["_delayed_count"] / kpis["vessel_calls"] * 100
    ).round(2)

    # avg_moves_per_crane_hour: pooled rate for the terminal-day
    safe_crane = kpis["crane_utilization_hours"].replace(0, np.nan)
    kpis["avg_moves_per_crane_hour"] = (
        kpis["total_container_moves"] / safe_crane
    ).round(2)

    # storm_days_count: 1 when at least one call on this terminal-day had storm weather
    kpis["storm_days_count"] = kpis["_storm_flag_any"].astype(int)

    # Round and cast
    kpis["avg_arrival_delay_minutes"]   = kpis["avg_arrival_delay_minutes"].round(1)
    kpis["avg_departure_delay_minutes"] = kpis["avg_departure_delay_minutes"].round(1)
    kpis["total_crane_hours"]           = kpis["total_crane_hours"].round(2)
    kpis["crane_utilization_hours"]     = kpis["crane_utilization_hours"].round(2)
    kpis["total_container_moves"]       = kpis["total_container_moves"].fillna(0).astype(int)

    # Final column order
    output_cols = [
        "operation_date",
        "terminal_id",
        "terminal_name",
        "vessel_calls",
        "avg_arrival_delay_minutes",
        "avg_departure_delay_minutes",
        "delayed_vessel_percentage",
        "total_container_moves",
        "total_crane_hours",
        "avg_moves_per_crane_hour",
        "crane_utilization_hours",
        "data_quality_issue_count",
        "storm_days_count",
    ]
    return kpis[output_cols].sort_values(
        ["operation_date", "terminal_id"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gold table 2: vessel call summary
# ---------------------------------------------------------------------------


def build_vessel_call_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Shape the enriched vessel-call frame into the gold vessel call summary.

    One row per vessel call with the requested columns:
        vessel_call_id, terminal_id, terminal_name, service_code,
        eta, ata, etd, atd,
        arrival_delay_minutes, departure_delay_minutes,
        planned_moves, actual_moves, total_crane_hours,
        moves_per_crane_hour, storm_flag, delay_category.
    """
    output_cols = [
        "vessel_call_id",
        "terminal_id",
        "terminal_name",
        "service_code",
        "eta",
        "ata",
        "etd",
        "atd",
        "arrival_delay_minutes",
        "departure_delay_minutes",
        "planned_moves",
        "actual_moves",
        "total_crane_hours",
        "moves_per_crane_hour",
        "storm_flag",
        "delay_category",
    ]

    df = enriched[output_cols].copy()

    df["arrival_delay_minutes"]   = df["arrival_delay_minutes"].round(1)
    df["departure_delay_minutes"] = df["departure_delay_minutes"].round(1)
    df["total_crane_hours"]       = df["total_crane_hours"].round(2)
    df["planned_moves"]           = df["planned_moves"].fillna(0).astype(int)
    df["actual_moves"]            = df["actual_moves"].fillna(0).astype(int)

    return df.sort_values("vessel_call_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def _print_summary(
    daily_kpis: pd.DataFrame,
    call_summary: pd.DataFrame,
) -> None:
    print()
    print("  Gold tables written")
    print(f"  {'Table':<40} {'Rows':>7}  {'Date range'}")
    print("  " + "-" * 75)

    date_lo = daily_kpis["operation_date"].min().strftime("%Y-%m-%d")
    date_hi = daily_kpis["operation_date"].max().strftime("%Y-%m-%d")
    print(f"  {'gold_daily_terminal_kpis':<40} {len(daily_kpis):>7,}  {date_lo} → {date_hi}")
    print(f"  {'gold_vessel_call_summary':<40} {len(call_summary):>7,}")

    print()
    print("  Top 10 terminal-days by vessel volume")
    print("  " + "-" * 75)
    top = (
        daily_kpis.nlargest(10, "vessel_calls")[
            ["operation_date", "terminal_id", "terminal_name",
             "vessel_calls", "delayed_vessel_percentage",
             "avg_arrival_delay_minutes", "storm_days_count"]
        ]
    )
    for _, row in top.iterrows():
        storm = " ⚠ storm" if row["storm_days_count"] else ""
        print(
            f"  {str(row['operation_date'])[:10]}  {row['terminal_id']:<5}  "
            f"{row['terminal_name']:<35}  calls={int(row['vessel_calls'])}  "
            f"delayed={row['delayed_vessel_percentage']:.0f}%  "
            f"arr_delay={row['avg_arrival_delay_minutes']:.0f} min{storm}"
        )

    print()
    print("  Delay category distribution (vessel call summary)")
    print("  " + "-" * 45)
    dist = call_summary["delay_category"].value_counts().sort_index()
    total = len(call_summary)
    for cat, n in dist.items():
        bar = "█" * int(n / total * 40)
        print(f"  {str(cat):<16} {n:>5,} ({n/total*100:4.1f}%)  {bar}")

    print()
    print(f"  Gold Parquet : {GOLD_DIR}")
    print(f"  Sample CSV   : {OUTPUTS_DIR / 'terminal_kpi_sample.csv'}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_dirs()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Gold KPI build  |  {ts}", flush=True)

    # Load silver
    log.info("Loading silver tables ...")
    tables = load_silver()
    vc  = tables["vessel_calls_silver"]
    ca  = tables["crane_assignments_silver"]
    mv  = tables["container_moves_silver"]
    wt  = tables["weather_daily_silver"]
    tm  = tables["terminal_metadata_silver"]

    # Pre-aggregate at vessel_call_id level
    log.info("Aggregating crane metrics per vessel call ...")
    ca_agg = agg_crane_per_call(ca)

    log.info("Aggregating container move counts per vessel call ...")
    mv_agg = agg_moves_per_call(mv)

    # Enrich
    log.info("Enriching vessel calls with all dimensions ...")
    enriched = enrich_vessel_calls(vc, ca_agg, mv_agg, wt, tm)
    enriched = add_gold_metadata(enriched)

    # Build gold tables
    log.info("Building gold_daily_terminal_kpis ...")
    daily_kpis = build_daily_terminal_kpis(enriched)
    daily_kpis = add_gold_metadata(daily_kpis)
    write_gold(daily_kpis, "gold_daily_terminal_kpis")

    log.info("Building gold_vessel_call_summary ...")
    call_summary = build_vessel_call_summary(enriched)
    call_summary = add_gold_metadata(call_summary)
    write_gold(call_summary, "gold_vessel_call_summary")

    # Write sample CSV
    sample = (
        daily_kpis
        .sort_values(["terminal_id", "operation_date"])
        .head(60)
    )
    sample.to_csv(OUTPUTS_DIR / "terminal_kpi_sample.csv", index=False)

    _print_summary(daily_kpis, call_summary)


if __name__ == "__main__":
    main()
