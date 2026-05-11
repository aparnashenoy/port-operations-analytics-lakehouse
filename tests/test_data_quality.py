"""
Integration tests against the live Silver and Gold Parquet files.

These tests validate that pipeline invariants hold end-to-end after the full
pipeline has run (generate_data → bronze_ingestion → silver_transformations
→ gold_kpis → feature_engineering).  They intentionally read from disk so
that any regression in the pipeline output is caught without re-running the
entire suite.

Run: pytest tests/test_data_quality.py
"""

import json
import pandas as pd
import pytest
from pathlib import Path

SILVER_DIR  = Path("data/silver")
GOLD_DIR    = Path("data/gold")
OUTPUTS_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Fixtures — load each file once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def silver_vessel_calls() -> pd.DataFrame:
    path = SILVER_DIR / "vessel_calls_silver.parquet"
    if not path.exists():
        pytest.skip(f"Silver data not found at {path} — run the pipeline first")
    return pd.read_parquet(path, engine="pyarrow")


@pytest.fixture(scope="module")
def silver_crane_assignments() -> pd.DataFrame:
    path = SILVER_DIR / "crane_assignments_silver.parquet"
    if not path.exists():
        pytest.skip(f"Silver data not found at {path} — run the pipeline first")
    return pd.read_parquet(path, engine="pyarrow")


@pytest.fixture(scope="module")
def gold_terminal_kpis() -> pd.DataFrame:
    path = GOLD_DIR / "gold_daily_terminal_kpis.parquet"
    if not path.exists():
        pytest.skip(f"Gold KPI table not found at {path} — run gold_kpis.py first")
    return pd.read_parquet(path, engine="pyarrow")


@pytest.fixture(scope="module")
def ml_features() -> pd.DataFrame:
    path = GOLD_DIR / "ml_vessel_delay_features.parquet"
    if not path.exists():
        pytest.skip(f"ML features not found at {path} — run feature_engineering.py first")
    return pd.read_parquet(path, engine="pyarrow")


# ---------------------------------------------------------------------------
# Silver vessel calls
# ---------------------------------------------------------------------------


def test_no_duplicate_vessel_call_ids(silver_vessel_calls):
    """
    Silver deduplication must leave exactly one row per vessel_call_id.

    Bronze deliberately contains late re-transmissions.  If any duplicates
    remain after silver, the deduplication step failed silently.
    """
    n_dupes = silver_vessel_calls["vessel_call_id"].duplicated().sum()
    assert n_dupes == 0, (
        f"{n_dupes} duplicate vessel_call_id values remain in silver after deduplication"
    )


# Every column in this list must be present after the silver transformation.
# Adding a column here is a contract: any future refactor that removes it
# will fail this test, forcing an explicit decision.
REQUIRED_SILVER_COLUMNS = [
    "vessel_call_id",
    "vessel_id",
    "terminal_id",
    "eta",
    "ata",
    "etd",
    "atd",
    "arrival_delay_minutes",
    "departure_delay_minutes",
    "missing_eta_flag",
    "missing_ata_flag",
    "invalid_arrival_sequence_flag",
    "invalid_departure_sequence_flag",
    "large_delay_outlier_flag",
    "_silver_ts",
]


@pytest.mark.parametrize("col", REQUIRED_SILVER_COLUMNS)
def test_required_silver_column_exists(col, silver_vessel_calls):
    """Every expected silver column must be present after transformation."""
    assert col in silver_vessel_calls.columns, (
        f"Required silver column '{col}' is missing — was it renamed or dropped?"
    )


def test_missing_eta_flag_matches_null_eta(silver_vessel_calls):
    """
    missing_eta_flag must be True for every row where eta is NULL and False
    elsewhere.  A mismatch means the flag was set incorrectly.
    """
    vc = silver_vessel_calls
    expected = vc["eta"].isna()
    actual   = vc["missing_eta_flag"]
    mismatched = (expected != actual).sum()
    assert mismatched == 0, (
        f"{mismatched} rows have mismatched missing_eta_flag vs actual eta nullity"
    )


def test_missing_ata_flag_matches_null_ata(silver_vessel_calls):
    """missing_ata_flag must agree with whether ata is NULL."""
    vc = silver_vessel_calls
    expected = vc["ata"].isna()
    mismatched = (expected != vc["missing_ata_flag"]).sum()
    assert mismatched == 0, (
        f"{mismatched} rows have mismatched missing_ata_flag vs actual ata nullity"
    )


# ---------------------------------------------------------------------------
# Silver crane assignments
# ---------------------------------------------------------------------------


def test_valid_crane_records_have_end_after_start(silver_crane_assignments):
    """
    Rows not flagged as invalid_crane_time must have actual_end >= actual_start.
    This verifies the flag logic in add_crane_flags() was correctly applied.
    """
    ca = silver_crane_assignments
    valid = ca[~ca["invalid_crane_time_flag"]]
    both_present = valid["actual_start"].notna() & valid["actual_end"].notna()
    violations = valid.loc[both_present & (valid["actual_end"] < valid["actual_start"])]
    assert len(violations) == 0, (
        f"{len(violations)} crane rows flagged as valid but have actual_end < actual_start"
    )


def test_invalid_crane_time_flag_covers_inverted_timestamps(silver_crane_assignments):
    """
    Every row where actual_end < actual_start must be flagged as invalid.
    This is the inverse of the check above — neither direction should be missed.
    """
    ca = silver_crane_assignments
    both_present = ca["actual_start"].notna() & ca["actual_end"].notna()
    inverted = ca.loc[both_present & (ca["actual_end"] < ca["actual_start"])]
    unflagged_inversions = inverted[~inverted["invalid_crane_time_flag"]]
    assert len(unflagged_inversions) == 0, (
        f"{len(unflagged_inversions)} crane rows have actual_end < actual_start but are NOT flagged"
    )


# ---------------------------------------------------------------------------
# Gold KPI grain
# ---------------------------------------------------------------------------


def test_gold_terminal_kpi_grain_unique(gold_terminal_kpis):
    """
    Gold daily_terminal_kpis must have exactly one row per
    (terminal_id, operation_date).  Duplicate grains would double-count KPIs
    in any dashboard or downstream aggregation.
    """
    n_dupes = gold_terminal_kpis.duplicated(
        subset=["terminal_id", "operation_date"]
    ).sum()
    assert n_dupes == 0, (
        f"{n_dupes} duplicate (terminal_id, operation_date) rows found in gold KPI table"
    )


def test_gold_terminal_kpi_has_five_terminals(gold_terminal_kpis):
    """Every one of the five synthetic terminals must appear in the KPI table."""
    expected_terminals = {"CPT", "EFT", "NCT", "SLH", "WIT"}
    actual_terminals   = set(gold_terminal_kpis["terminal_id"].unique())
    missing = expected_terminals - actual_terminals
    assert not missing, f"Terminals absent from gold KPI table: {missing}"


# ---------------------------------------------------------------------------
# ML feature table — schema and leakage
# ---------------------------------------------------------------------------


REQUIRED_ML_FEATURES = [
    "terminal_id_encoded",
    "service_code_encoded",
    "vessel_capacity_teu",
    "planned_moves",
    "planned_crane_count",
    "planned_crane_hours",
    "day_of_week",
    "month",
    "is_weekend",
    "storm_flag",
    "avg_previous_10_terminal_delays",
    "avg_previous_10_service_delays",
    "previous_vessel_delay",
    "terminal_congestion_score",
]


@pytest.mark.parametrize("feature", REQUIRED_ML_FEATURES)
def test_ml_feature_table_contains_required_feature(feature, ml_features):
    """All 14 model input features must be present in the ML feature table."""
    assert feature in ml_features.columns, (
        f"Required ML feature '{feature}' is missing from the feature table"
    )


def test_ml_feature_table_has_target_column(ml_features):
    """The binary target column must be present; it is the outcome the model predicts."""
    assert "is_arrival_delayed_more_than_2_hours" in ml_features.columns


# Columns that contain post-arrival actuals — must NOT appear in the feature table.
# If they appear, the feature pipeline is leaking future information.
LEAKAGE_COLUMNS = [
    "ata",
    "atd",
    "arrival_delay_minutes",
    "departure_delay_minutes",
    "delay_hours",
    "actual_turnaround_hours",
]


@pytest.mark.parametrize("col", LEAKAGE_COLUMNS)
def test_leakage_column_absent_from_ml_feature_table(col, ml_features):
    """
    Post-arrival actuals must not appear in the ML feature table.

    These values are only known after the vessel arrives — exactly the event
    we are trying to predict.  Their presence would give the model access to
    future information and produce inflated metrics that fail in production.
    """
    assert col not in ml_features.columns, (
        f"Leakage column '{col}' found in the ML feature table. "
        "It is a post-arrival actual and must be excluded from the feature set."
    )


# ---------------------------------------------------------------------------
# DQ report — regression guard
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dq_report() -> dict:
    """
    Load the structured DQ report produced by data_quality_checks.py.

    The report is the authoritative record of pipeline health.  These tests
    lock in the expected check outcomes so that any new FAIL introduced by a
    code change is caught immediately rather than discovered in production.
    """
    path = OUTPUTS_DIR / "data_quality_report.json"
    if not path.exists():
        pytest.skip(f"DQ report not found at {path} — run data_quality_checks.py first")
    with open(path) as f:
        return json.load(f)


def test_dq_report_has_expected_check_count(dq_report):
    """The report must contain exactly 14 checks — one per defined rule."""
    n = dq_report["summary"]["total_checks"]
    assert n == 14, f"Expected 14 DQ checks, found {n}"


def test_dq_report_exactly_one_fail(dq_report):
    """
    Exactly one check should be in FAIL status: crane_overlaps (37.7%,
    above the 15% threshold).  This failure is intentional — it represents
    a real operational failure mode injected into the synthetic data to
    demonstrate that the framework correctly classifies it.

    A second FAIL here means a new regression was introduced.
    """
    failed = [k for k, v in dq_report["checks"].items() if v["status"] == "FAIL"]
    assert failed == ["crane_overlaps"], (
        f"Expected exactly ['crane_overlaps'] to fail. Got: {failed}"
    )


def test_dq_report_silver_deduplication_passes(dq_report):
    """Silver deduplication must leave zero duplicate vessel_call_ids."""
    status = dq_report["checks"]["silver_duplicate_vessel_calls"]["status"]
    assert status == "PASS", f"silver_duplicate_vessel_calls is {status}, expected PASS"


def test_dq_report_gold_grain_checks_pass(dq_report):
    """Both gold grain checks (call summary and terminal KPI) must pass."""
    for check in ("gold_call_summary_grain", "gold_terminal_kpi_grain"):
        status = dq_report["checks"][check]["status"]
        assert status == "PASS", f"{check} is {status}, expected PASS"


def test_dq_report_ml_feature_nulls_pass(dq_report):
    """ML feature null-rate check must pass (all 14 feature columns clean)."""
    status = dq_report["checks"]["ml_feature_missing_values"]["status"]
    assert status == "PASS", f"ml_feature_missing_values is {status}, expected PASS"


def test_dq_report_cross_layer_counts_pass(dq_report):
    """
    ML feature table must have fewer distinct call IDs than silver.
    A ML > silver count would indicate fabricated rows — a data integrity failure.
    """
    status = dq_report["checks"]["cross_layer_call_id_counts"]["status"]
    assert status == "PASS", f"cross_layer_call_id_counts is {status}, expected PASS"
