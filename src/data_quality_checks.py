"""
Data quality validation suite for the port operations analytics lakehouse.

Reads Silver and Gold Parquet files and runs 14 targeted checks covering
completeness, timestamp consistency, referential integrity, grain uniqueness,
and ML feature readiness.  Results are graded and written to
outputs/data_quality_report.json.

Check grades
  PASS  — value meets the expectation exactly or within tolerance
  WARN  — value exceeds an acceptable threshold; warrants investigation
  FAIL  — hard expectation violated; downstream outputs are unreliable

Run:
    python src/data_quality_checks.py
"""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    GOLD_DIR,
    OUTPUTS_DIR,
    SILVER_DIR,
    ensure_dirs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Check result container
# ---------------------------------------------------------------------------

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_SEVERITY = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass
class CheckResult:
    name:        str
    description: str
    status:      str   # PASS | WARN | FAIL
    value:       Any   # measured value — int, float, dict, etc.
    threshold:   Any   # expectation or bound for comparison
    detail:      Any = None  # supplementary information (optional)

    def as_dict(self) -> dict:
        return {
            "status":      self.status,
            "description": self.description,
            "value":       self.value,
            "threshold":   self.threshold,
            "detail":      self.detail,
        }


def _grade(value: float, warn_at: float, fail_at: Optional[float] = None) -> str:
    """Return PASS / WARN / FAIL by comparing value against thresholds."""
    if fail_at is not None and value > fail_at:
        return FAIL
    if value > warn_at:
        return WARN
    return PASS


def _worst(*statuses: str) -> str:
    return max(statuses, key=lambda s: _SEVERITY[s])


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_tables() -> dict[str, pd.DataFrame]:
    """Load all Silver and Gold tables required for the checks."""
    paths = {
        "vessel_calls":      SILVER_DIR / "vessel_calls_silver.parquet",
        "crane_assignments": SILVER_DIR / "crane_assignments_silver.parquet",
        "call_summary":      GOLD_DIR   / "gold_vessel_call_summary.parquet",
        "terminal_kpis":     GOLD_DIR   / "gold_daily_terminal_kpis.parquet",
        "ml_features":       GOLD_DIR   / "ml_vessel_delay_features.parquet",
    }
    tables: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}. "
                "Run the full pipeline first (silver_transformations → gold_kpis → feature_engineering)."
            )
        tables[name] = pd.read_parquet(path, engine="pyarrow")
        log.info("Loaded %-22s  %d rows", name, len(tables[name]))
    return tables


# ---------------------------------------------------------------------------
# Silver vessel call checks (6 checks)
# ---------------------------------------------------------------------------


def check_duplicate_vessel_calls(vc: pd.DataFrame) -> CheckResult:
    """
    Silver deduplication removes duplicate vessel_call_id values introduced by
    late re-transmissions in bronze.  Any remaining duplicates mean the
    deduplication step did not run or silently failed.
    """
    n_dupes    = int(vc["vessel_call_id"].duplicated(keep=False).sum())
    sample_ids = (
        vc.loc[vc["vessel_call_id"].duplicated(keep=False), "vessel_call_id"]
        .unique().tolist()[:10]
    )
    return CheckResult(
        name="silver_duplicate_vessel_calls",
        description="Duplicate vessel_call_id values in silver after deduplication",
        status=FAIL if n_dupes > 0 else PASS,
        value=n_dupes,
        threshold=0,
        detail={"sample_ids": sample_ids} if n_dupes > 0 else None,
    )


def check_missing_eta(vc: pd.DataFrame) -> CheckResult:
    """
    Missing ETA is injected into bronze as a deliberate quality issue.
    After silver processing the rate should stay below 5 %; above 15 % it
    suggests a systematic upstream extraction failure.
    """
    n      = int(vc["missing_eta_flag"].sum())
    rate   = round(100 * n / len(vc), 2)
    return CheckResult(
        name="silver_missing_eta",
        description="Vessel calls where ETA is NULL (cannot be used as prediction targets)",
        status=_grade(rate, warn_at=5.0, fail_at=15.0),
        value={"count": n, "rate_pct": rate},
        threshold={"warn_pct": 5.0, "fail_pct": 15.0},
    )


def check_missing_ata(vc: pd.DataFrame) -> CheckResult:
    """
    Missing ATA is expected for in-progress calls; a rate above 10 % across the
    full historical table (not just recent calls) warrants investigation.
    """
    n      = int(vc["missing_ata_flag"].sum())
    rate   = round(100 * n / len(vc), 2)
    return CheckResult(
        name="silver_missing_ata",
        description="Vessel calls where ATA is NULL (in-progress or missing arrival record)",
        status=_grade(rate, warn_at=10.0, fail_at=25.0),
        value={"count": n, "rate_pct": rate},
        threshold={"warn_pct": 10.0, "fail_pct": 25.0},
    )


def check_invalid_arrival_sequence(vc: pd.DataFrame) -> CheckResult:
    """
    ATA > ATD is physically impossible: a vessel cannot depart before it arrives.
    Every occurrence is a data error, so any count above zero is a WARN.
    """
    n = int(vc["invalid_arrival_sequence_flag"].sum())
    return CheckResult(
        name="silver_invalid_arrival_sequence",
        description="Calls where ATA is recorded after ATD (physically impossible sequence)",
        status=WARN if n > 0 else PASS,
        value=n,
        threshold=0,
    )


def check_invalid_departure_sequence(vc: pd.DataFrame) -> CheckResult:
    """
    ETD < ETA means the planned departure is before the planned arrival —
    a scheduling entry error.  Any occurrence is flagged as WARN.
    """
    n = int(vc["invalid_departure_sequence_flag"].sum())
    return CheckResult(
        name="silver_invalid_departure_sequence",
        description="Calls where ETD is scheduled before ETA (impossible planned sequence)",
        status=WARN if n > 0 else PASS,
        value=n,
        threshold=0,
    )


def check_large_delay_outliers(vc: pd.DataFrame) -> CheckResult:
    """
    Records flagged by the Tukey IQR test (delay_hours > Q3 + 1.5×IQR, min 24 h).
    A small number of genuine extreme events is acceptable (PASS below 1 %).
    A high rate indicates systematic reporting issues rather than genuine delays.
    """
    n      = int(vc["large_delay_outlier_flag"].sum())
    rate   = round(100 * n / len(vc), 2)
    return CheckResult(
        name="silver_large_delay_outliers",
        description="Calls where delay_hours exceeds the Tukey upper fence (min 24 h)",
        status=_grade(rate, warn_at=1.0, fail_at=5.0),
        value={"count": n, "rate_pct": rate},
        threshold={"warn_pct": 1.0, "fail_pct": 5.0},
    )


# ---------------------------------------------------------------------------
# Silver crane assignment checks (3 checks)
# ---------------------------------------------------------------------------


def check_crane_end_before_start(ca: pd.DataFrame) -> CheckResult:
    """
    Crane assignments where actual_end < actual_start are invalid: a crane
    session cannot end before it began.  These are flagged in silver via
    invalid_crane_time_flag.  Rates above 2 % are unusual enough to warn.
    """
    n    = int(ca["invalid_crane_time_flag"].sum())
    rate = round(100 * n / len(ca), 2)
    return CheckResult(
        name="crane_end_before_start",
        description="Crane assignments where actual_end is earlier than actual_start",
        status=_grade(rate, warn_at=2.0, fail_at=10.0),
        value={"count": n, "rate_pct": rate},
        threshold={"warn_pct": 2.0, "fail_pct": 10.0},
    )


def check_crane_overlaps(ca: pd.DataFrame) -> CheckResult:
    """
    A physical crane cannot serve two calls simultaneously.  Overlapping
    assignments indicate scheduling conflicts or data entry errors.
    crane_overlap_flag is set on both parties of each conflicting pair.
    Above 5 % of rows suggests a systematic problem in the source schedule.
    """
    n                = int(ca["crane_overlap_flag"].sum())
    rate             = round(100 * n / len(ca), 2)
    affected_cranes  = int(ca.loc[ca["crane_overlap_flag"], "crane_id"].nunique())
    return CheckResult(
        name="crane_overlaps",
        description="Crane assignments that temporally overlap an earlier assignment for the same crane",
        status=_grade(rate, warn_at=5.0, fail_at=15.0),
        value={"count": n, "rate_pct": rate},
        threshold={"warn_pct": 5.0, "fail_pct": 15.0},
        detail={"affected_cranes": affected_cranes},
    )


def check_orphan_crane_assignments(ca: pd.DataFrame, vc: pd.DataFrame) -> CheckResult:
    """
    Every crane assignment must reference a vessel_call_id that exists in silver
    vessel_calls.  Orphaned assignments arise when a call is quarantined without
    cascading the removal to its related tables.
    """
    valid_ids    = set(vc["vessel_call_id"].dropna().astype(str))
    orphan_mask  = ~ca["vessel_call_id"].astype(str).isin(valid_ids)
    n            = int(orphan_mask.sum())
    rate         = round(100 * n / len(ca), 2)
    sample_ids   = ca.loc[orphan_mask, "vessel_call_id"].unique().tolist()[:10]
    return CheckResult(
        name="orphan_crane_assignments",
        description="Crane assignments referencing a vessel_call_id absent from silver vessel_calls",
        status=FAIL if n > 0 else PASS,
        value={"count": n, "rate_pct": rate},
        threshold=0,
        detail={"sample_orphan_call_ids": sample_ids} if n > 0 else None,
    )


# ---------------------------------------------------------------------------
# Gold layer grain checks (2 checks)
# ---------------------------------------------------------------------------


def check_call_summary_grain(gs: pd.DataFrame) -> CheckResult:
    """
    Gold vessel_call_summary is at vessel-call grain.  A duplicate vessel_call_id
    would cause incorrect joins downstream and indicates a fan-out bug in gold_kpis.py.
    """
    dupes = int(gs["vessel_call_id"].duplicated().sum())
    return CheckResult(
        name="gold_call_summary_grain",
        description="Gold vessel_call_summary grain: one row per vessel_call_id",
        status=FAIL if dupes > 0 else PASS,
        value={"total_rows": len(gs), "duplicate_rows": dupes},
        threshold=0,
    )


def check_terminal_kpi_grain(kpi: pd.DataFrame) -> CheckResult:
    """
    Gold daily_terminal_kpis is at (terminal_id, operation_date) grain.
    Duplicates would double-count KPI aggregates in any downstream dashboard.
    """
    dupes = int(kpi.duplicated(subset=["terminal_id", "operation_date"]).sum())
    return CheckResult(
        name="gold_terminal_kpi_grain",
        description="Gold daily_terminal_kpis grain: one row per (terminal_id, operation_date)",
        status=FAIL if dupes > 0 else PASS,
        value={"total_rows": len(kpi), "duplicate_rows": dupes},
        threshold=0,
    )


# ---------------------------------------------------------------------------
# ML feature table checks (2 checks)
# ---------------------------------------------------------------------------

# Columns expected to have zero nulls (fully static or derivable from ETA).
_ZERO_NULL_FEATURES = [
    "terminal_id_encoded", "service_code_encoded",
    "vessel_capacity_teu", "planned_moves", "planned_crane_count", "planned_crane_hours",
    "day_of_week", "month", "is_weekend", "storm_flag", "terminal_congestion_score",
]

# Columns that legitimately have nulls for calls with no historical precedent.
_HISTORY_FEATURES = [
    "avg_previous_10_terminal_delays",
    "avg_previous_10_service_delays",
    "previous_vessel_delay",
]


def check_ml_feature_missing_values(feat: pd.DataFrame) -> CheckResult:
    """
    Null-rate audit across all 14 model input columns.

    Static and temporal features (terminal, temporal, weather, scope) should
    have 0 % nulls — any null means the feature pipeline failed.  Rolling
    history features tolerate higher null rates because calls with no prior
    history in the simulation window have no qualifying records.
    """
    total = len(feat)
    per_column: dict[str, dict] = {}
    worst_status = PASS

    all_features = _ZERO_NULL_FEATURES + _HISTORY_FEATURES
    for col in all_features:
        if col not in feat.columns:
            per_column[col] = {"null_count": None, "null_rate_pct": None, "status": FAIL}
            worst_status = FAIL
            continue

        n_null   = int(feat[col].isna().sum())
        rate_pct = round(100 * n_null / total, 2)
        is_hist  = col in _HISTORY_FEATURES
        status   = _grade(rate_pct, warn_at=(15.0 if is_hist else 0.0),
                                    fail_at=(50.0 if is_hist else 5.0))
        worst_status = _worst(worst_status, status)
        per_column[col] = {
            "null_count":    n_null,
            "null_rate_pct": rate_pct,
            "status":        status,
        }

    columns_with_issues = {k: v for k, v in per_column.items() if v["status"] != PASS}

    return CheckResult(
        name="ml_feature_missing_values",
        description="Null rate per feature column in the ML feature table",
        status=worst_status,
        value={
            "columns_checked":       len(all_features),
            "columns_clean":         len(all_features) - len(columns_with_issues),
            "columns_with_nulls":    len(columns_with_issues),
        },
        threshold={
            "static_features_fail_pct":  5.0,
            "history_features_fail_pct": 50.0,
        },
        detail={"per_column": per_column},
    )


def check_ml_target_distribution(feat: pd.DataFrame) -> CheckResult:
    """
    Binary target distribution sanity check.

    A positive rate outside [5 %, 60 %] suggests misconfigured labelling logic
    or a synthetic data generation bug.  The 19 unlabeled rows (in-progress
    calls with no recorded ATA) are reported separately and are expected.
    """
    target_col = "is_arrival_delayed_more_than_2_hours"
    total      = len(feat)
    labeled    = int(feat[target_col].notna().sum())
    unlabeled  = total - labeled
    positive   = int((feat[target_col] == 1.0).sum())
    negative   = int((feat[target_col] == 0.0).sum())
    pos_rate   = round(100 * positive / labeled, 2) if labeled > 0 else 0.0

    # Grade: suspicious if almost all one class
    status = _grade(pos_rate, warn_at=60.0, fail_at=95.0)
    if pos_rate < 5.0:
        # Extremely rare positives → model has almost no signal to learn
        status = _worst(status, WARN)

    return CheckResult(
        name="ml_target_distribution",
        description="Binary target distribution in the ML feature table",
        status=status,
        value={
            "total_rows":        total,
            "labeled_rows":      labeled,
            "unlabeled_rows":    unlabeled,
            "positive_count":    positive,
            "negative_count":    negative,
            "positive_rate_pct": pos_rate,
        },
        threshold={"warn_if_below_pct": 5.0, "warn_if_above_pct": 60.0, "fail_if_above_pct": 95.0},
    )


# ---------------------------------------------------------------------------
# Cross-layer reconciliation (1 check)
# ---------------------------------------------------------------------------


def check_cross_layer_counts(
    vc: pd.DataFrame,
    gs: pd.DataFrame,
    feat: pd.DataFrame,
) -> CheckResult:
    """
    Distinct vessel_call_id counts must be consistent across layers:
      - Silver == Gold summary (both cover all 2000 deduplicated calls).
      - ML features <= Silver (valid ETA + no invalid timestamps is a subset).
      - ML features > Silver would be a serious data error (fabricated rows).
    """
    silver_ids = int(vc["vessel_call_id"].nunique())
    gold_ids   = int(gs["vessel_call_id"].nunique())
    ml_ids     = int(feat["vessel_call_id"].nunique())

    gold_ok = gold_ids == silver_ids
    ml_ok   = ml_ids <= silver_ids

    status = PASS
    if not gold_ok:
        status = _worst(status, WARN)
    if ml_ids > silver_ids:
        # More IDs in ML table than silver — impossible without fabrication
        status = FAIL

    return CheckResult(
        name="cross_layer_call_id_counts",
        description="Distinct vessel_call_id counts across silver, gold summary, and ML feature layers",
        status=status,
        value={
            "silver_distinct_ids":   silver_ids,
            "gold_summary_ids":      gold_ids,
            "ml_feature_ids":        ml_ids,
        },
        threshold={
            "gold_should_equal_silver":          True,
            "ml_should_be_subset_of_silver":     True,
        },
    )


# ---------------------------------------------------------------------------
# Report assembly and output
# ---------------------------------------------------------------------------


def assemble_report(results: list[CheckResult]) -> dict:
    """Aggregate all CheckResult objects into the final JSON payload."""
    passed = sum(1 for r in results if r.status == PASS)
    warned = sum(1 for r in results if r.status == WARN)
    failed = sum(1 for r in results if r.status == FAIL)

    overall = PASS
    if failed > 0:
        overall = FAIL
    elif warned > 0:
        overall = WARN

    return {
        "run_timestamp":  datetime.now(tz=timezone.utc).isoformat(),
        "overall_status": overall,
        "summary": {
            "total_checks": len(results),
            "passed":       passed,
            "warned":       warned,
            "failed":       failed,
        },
        "checks": {r.name: r.as_dict() for r in results},
    }


def save_report(report: dict) -> Path:
    out = OUTPUTS_DIR / "data_quality_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    log.info("Report written to %s", out)
    return out


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------


def _fmt_value(check: dict) -> str:
    """Render a check's value field as a compact string for the console table."""
    v = check["value"]
    if isinstance(v, dict):
        if "count" in v and "rate_pct" in v:
            return f"{v['count']} rows ({v['rate_pct']}%)"
        if "duplicate_rows" in v:
            return f"{v['duplicate_rows']} dupes / {v['total_rows']} rows"
        if "columns_checked" in v:
            n_issues = v.get("columns_with_nulls", 0)
            return f"{n_issues} of {v['columns_checked']} columns have nulls"
        if "positive_rate_pct" in v:
            return (
                f"{v['positive_count']} positive / {v['labeled_rows']} labeled "
                f"({v['positive_rate_pct']}%)"
            )
        if "silver_distinct_ids" in v:
            return (
                f"silver={v['silver_distinct_ids']}  "
                f"gold={v['gold_summary_ids']}  "
                f"ml={v['ml_feature_ids']}"
            )
    return str(v)


def print_report(report: dict) -> None:
    s = report["summary"]
    overall = report["overall_status"]

    print()
    print(f"  Overall  {overall}  "
          f"({s['total_checks']} checks: "
          f"{s['passed']} passed  {s['warned']} warned  {s['failed']} failed)")
    print()
    print(f"  {'Check':<42}  {'Status':<6}  Value")
    print("  " + "─" * 90)
    for name, check in report["checks"].items():
        status  = check["status"]
        val_str = _fmt_value(check)
        label   = name.replace("_", " ")
        print(f"  {label:<42}  {status:<6}  {val_str}")
    print()

    # Surface any WARN/FAIL details
    issues = {n: c for n, c in report["checks"].items() if c["status"] != PASS}
    if issues:
        print("  Details for checks that did not pass:")
        for name, check in issues.items():
            print(f"    [{check['status']}] {name}")
            print(f"      {check['description']}")
            if check.get("detail"):
                # Print detail sub-keys compactly (skip per_column — it's large)
                detail = check["detail"]
                if isinstance(detail, dict) and "per_column" in detail:
                    # Show only columns that are not clean
                    issues_cols = {
                        col: info for col, info in detail["per_column"].items()
                        if info.get("status") != PASS
                    }
                    if issues_cols:
                        for col, info in issues_cols.items():
                            print(f"        {col}: {info['null_count']} nulls "
                                  f"({info['null_rate_pct']}%) [{info['status']}]")
                else:
                    print(f"        {detail}")
        print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_dirs()

    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Data quality checks  |  {ts}", flush=True)

    log.info("Loading tables ...")
    tables = load_tables()
    vc   = tables["vessel_calls"]
    ca   = tables["crane_assignments"]
    gs   = tables["call_summary"]
    kpi  = tables["terminal_kpis"]
    feat = tables["ml_features"]

    log.info("Running %d checks ...", 14)

    results = [
        # Silver vessel calls (6 checks)
        check_duplicate_vessel_calls(vc),
        check_missing_eta(vc),
        check_missing_ata(vc),
        check_invalid_arrival_sequence(vc),
        check_invalid_departure_sequence(vc),
        check_large_delay_outliers(vc),
        # Silver crane assignments (3 checks)
        check_crane_end_before_start(ca),
        check_crane_overlaps(ca),
        check_orphan_crane_assignments(ca, vc),
        # Gold grain uniqueness (2 checks)
        check_call_summary_grain(gs),
        check_terminal_kpi_grain(kpi),
        # ML feature table (2 checks)
        check_ml_feature_missing_values(feat),
        check_ml_target_distribution(feat),
        # Cross-layer reconciliation (1 check)
        check_cross_layer_counts(vc, gs, feat),
    ]

    report = assemble_report(results)
    save_report(report)
    print_report(report)


if __name__ == "__main__":
    main()
