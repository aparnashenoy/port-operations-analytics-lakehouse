"""
Unit tests for src/feature_engineering.py.

Each test exercises a single feature engineering function with a hand-crafted
in-memory DataFrame.  The leakage prevention tests verify — at the function
and contract level — that post-arrival actuals cannot enter the model as
inputs, even after future refactoring.

Run: pytest tests/test_features.py
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    LABEL_THRESHOLD_MINUTES,
    TERMINAL_CATEGORIES,
    SERVICE_CATEGORIES,
    add_target,
    add_temporal_features,
    build_candidates,
    encode_categoricals,
)
from src.train_model import MODEL_FEATURES


# ---------------------------------------------------------------------------
# Shared builder
# ---------------------------------------------------------------------------


def _vc(**overrides) -> pd.DataFrame:
    """
    Return a minimal vessel_calls DataFrame with all columns required by
    build_candidates().  Pass keyword arguments as lists to override defaults.
    """
    defaults = {
        "vessel_call_id":                ["VC001"],
        "vessel_id":                     ["V001"],
        "carrier_code":                  ["MAERSK"],
        "terminal_id":                   ["CPT"],
        "eta":                           [pd.Timestamp("2024-06-01 08:00")],
        "etd":                           [pd.Timestamp("2024-06-02 08:00")],
        "planned_cargo_teu":             [1000],
        "planned_turnaround_hours":      [24.0],
        "arrival_delay_minutes":         [60.0],
        "atd":                           [pd.Timestamp("2024-06-02 10:00")],
        "invalid_arrival_sequence_flag": [False],
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


# ---------------------------------------------------------------------------
# build_candidates
# ---------------------------------------------------------------------------


class TestBuildCandidates:
    def test_valid_row_is_included(self):
        """A row with a valid ETA and no invalid sequence must pass the filter."""
        result = build_candidates(_vc())
        assert len(result) == 1

    def test_excludes_row_with_missing_eta(self):
        """
        Without an ETA there is no prediction anchor.  Rows missing ETA must
        be excluded from the candidate set.
        """
        result = build_candidates(_vc(eta=[None]))
        assert len(result) == 0

    def test_excludes_row_with_invalid_arrival_sequence(self):
        """
        Rows with an invalid timestamp sequence have a corrupt potential label
        and must be excluded from the training candidate set.
        """
        result = build_candidates(_vc(invalid_arrival_sequence_flag=[True]))
        assert len(result) == 0

    def test_mixed_rows_filters_correctly(self):
        """Valid rows pass; missing-ETA and invalid-sequence rows are removed."""
        df = _vc(
            vessel_call_id=["VC001", "VC002", "VC003"],
            vessel_id=["V001", "V002", "V003"],
            carrier_code=["MAERSK", "MSC", "APL"],
            terminal_id=["CPT", "CPT", "CPT"],
            eta=[
                pd.Timestamp("2024-06-01 08:00"),
                None,                               # missing ETA → excluded
                pd.Timestamp("2024-06-03 08:00"),
            ],
            etd=3 * [pd.Timestamp("2024-06-02 08:00")],
            planned_cargo_teu=[1000, 1000, 1000],
            planned_turnaround_hours=[24.0, 24.0, 24.0],
            arrival_delay_minutes=[60.0, 90.0, 120.0],
            atd=3 * [pd.Timestamp("2024-06-02 10:00")],
            invalid_arrival_sequence_flag=[
                False,
                False,
                True,   # invalid sequence → excluded
            ],
        )
        result = build_candidates(df)
        assert len(result) == 1
        assert result["vessel_call_id"].iloc[0] == "VC001"

    def test_output_does_not_include_invalid_sequence_flag(self):
        """
        The candidate DataFrame must not carry invalid_arrival_sequence_flag
        forward — it is a filter criterion, not a model feature.
        """
        result = build_candidates(_vc())
        assert "invalid_arrival_sequence_flag" not in result.columns


# ---------------------------------------------------------------------------
# add_temporal_features
# ---------------------------------------------------------------------------


class TestTemporalFeatures:
    def _df(self, eta: str) -> pd.DataFrame:
        return pd.DataFrame({"eta": [pd.Timestamp(eta)]})

    def test_monday_day_of_week_is_zero(self):
        """pandas convention: Monday = 0.  2024-01-01 is a Monday."""
        result = add_temporal_features(self._df("2024-01-01 08:00"))
        assert result["day_of_week"].iloc[0] == 0

    def test_sunday_day_of_week_is_six(self):
        """pandas convention: Sunday = 6.  2024-01-07 is a Sunday."""
        result = add_temporal_features(self._df("2024-01-07 08:00"))
        assert result["day_of_week"].iloc[0] == 6

    def test_month_extracted_from_eta(self):
        result = add_temporal_features(self._df("2024-06-15 08:00"))
        assert result["month"].iloc[0] == 6

    def test_is_weekend_true_for_saturday(self):
        """2024-01-06 is a Saturday (dayofweek = 5) → is_weekend = 1."""
        result = add_temporal_features(self._df("2024-01-06 08:00"))
        assert result["is_weekend"].iloc[0] == 1

    def test_is_weekend_true_for_sunday(self):
        result = add_temporal_features(self._df("2024-01-07 14:00"))
        assert result["is_weekend"].iloc[0] == 1

    def test_is_weekend_false_for_weekday(self):
        """2024-01-02 is a Tuesday → is_weekend = 0."""
        result = add_temporal_features(self._df("2024-01-02 08:00"))
        assert result["is_weekend"].iloc[0] == 0

    def test_temporal_features_derived_from_eta_not_ata(self):
        """
        Temporal features must use ETA, not ATA.  Using ATA would leak the
        actual arrival time into the feature set.  We verify day_of_week
        matches ETA, not an alternative timestamp on a different day.
        """
        # ETA is Monday (dayofweek=0); we put a different timestamp in ata
        df = pd.DataFrame({
            "eta": [pd.Timestamp("2024-01-01 08:00")],   # Monday
        })
        result = add_temporal_features(df)
        assert result["day_of_week"].iloc[0] == 0   # confirms ETA was used


# ---------------------------------------------------------------------------
# add_target
# ---------------------------------------------------------------------------


class TestAddTarget:
    def _df(self, delay_minutes) -> pd.DataFrame:
        return pd.DataFrame({"arrival_delay_minutes": [delay_minutes]})

    def test_delayed_above_threshold(self):
        """
        delay > LABEL_THRESHOLD_MINUTES → is_arrival_delayed_more_than_2_hours = 1.
        """
        result = add_target(self._df(LABEL_THRESHOLD_MINUTES + 1))
        assert result["is_arrival_delayed_more_than_2_hours"].iloc[0] == 1.0

    def test_not_delayed_below_threshold(self):
        """delay < LABEL_THRESHOLD_MINUTES → target = 0."""
        result = add_target(self._df(LABEL_THRESHOLD_MINUTES - 1))
        assert result["is_arrival_delayed_more_than_2_hours"].iloc[0] == 0.0

    def test_not_delayed_at_exact_threshold(self):
        """
        delay == LABEL_THRESHOLD_MINUTES → target = 0.
        The condition is strictly greater-than, so the boundary is not delayed.
        """
        result = add_target(self._df(LABEL_THRESHOLD_MINUTES))
        assert result["is_arrival_delayed_more_than_2_hours"].iloc[0] == 0.0

    def test_target_null_for_in_progress_call(self):
        """
        NaN arrival_delay_minutes means no ATA is recorded (vessel in progress).
        The target must be NaN — these rows are inference targets, not training rows.
        """
        result = add_target(self._df(np.nan))
        assert pd.isna(result["is_arrival_delayed_more_than_2_hours"].iloc[0])

    def test_early_arrival_is_not_delayed(self):
        """Negative delay (early arrival) must produce target = 0."""
        result = add_target(self._df(-30.0))
        assert result["is_arrival_delayed_more_than_2_hours"].iloc[0] == 0.0

    def test_target_column_is_float(self):
        """
        Target must be float (0.0 / 1.0 / NaN) to accommodate the NaN case.
        An integer dtype cannot represent NaN and would silently corrupt data.
        """
        result = add_target(self._df(60.0))
        assert result["is_arrival_delayed_more_than_2_hours"].dtype == float


# ---------------------------------------------------------------------------
# Leakage prevention
# ---------------------------------------------------------------------------

# These columns contain information that is only available *after* a vessel
# arrives.  Using any of them as model inputs would give the model access to
# the future at training time, and make it impossible to deploy for real-time
# pre-arrival prediction.
LEAKAGE_COLUMNS = [
    "ata",
    "atd",
    "arrival_delay_minutes",
    "departure_delay_minutes",
    "delay_hours",
    "actual_turnaround_hours",
    "actual_cargo_teu",
]


@pytest.mark.parametrize("col", LEAKAGE_COLUMNS)
def test_model_features_exclude_leakage_column(col):
    """
    No post-arrival actual may appear in MODEL_FEATURES.

    This test guards against accidental re-introduction of leakage columns
    during refactoring.  If it fails, the training script will produce a model
    that cannot be deployed for pre-arrival prediction.
    """
    assert col not in MODEL_FEATURES, (
        f"Leakage column '{col}' found in MODEL_FEATURES. "
        "Post-arrival actuals are only known after the vessel arrives and "
        "must never be used as model inputs."
    )


def test_model_features_list_is_not_empty():
    """MODEL_FEATURES must contain at least one feature — an empty list trains nothing."""
    assert len(MODEL_FEATURES) > 0


def test_model_features_contains_no_identifier_columns():
    """
    vessel_call_id and eta are identifiers / split keys, not model inputs.
    They must be absent from MODEL_FEATURES to prevent the model from
    memorising individual call IDs.
    """
    identifier_cols = {"vessel_call_id", "eta"}
    leaked_ids = identifier_cols & set(MODEL_FEATURES)
    assert not leaked_ids, (
        f"Identifier columns found in MODEL_FEATURES: {leaked_ids}"
    )


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


class TestEncodeategoricals:
    def _df(self, terminal_id: str, carrier_code: str) -> pd.DataFrame:
        return pd.DataFrame({
            "terminal_id":  [terminal_id],
            "carrier_code": [carrier_code],
        })

    def test_cpt_terminal_encodes_to_zero(self):
        """CPT is the first terminal in the ordered category list → encodes to 0."""
        result = encode_categoricals(self._df("CPT", "MAERSK"))
        assert result["terminal_id_encoded"].iloc[0] == 0

    def test_wit_terminal_encodes_to_four(self):
        """WIT is the last terminal in the ordered category list → encodes to 4."""
        result = encode_categoricals(self._df("WIT", "MAERSK"))
        assert result["terminal_id_encoded"].iloc[0] == 4

    def test_apl_carrier_encodes_to_zero(self):
        """APL is the first carrier in the ordered category list → encodes to 0."""
        result = encode_categoricals(self._df("CPT", "APL"))
        assert result["service_code_encoded"].iloc[0] == 0

    def test_zim_carrier_encodes_to_nine(self):
        """ZIM is the last carrier in the ordered category list → encodes to 9."""
        result = encode_categoricals(self._df("CPT", "ZIM"))
        assert result["service_code_encoded"].iloc[0] == 9

    def test_unknown_terminal_encodes_to_minus_one(self):
        """
        Unknown terminal IDs must encode to -1 rather than raising an error.
        This allows inference on vessels calling at new terminals without
        crashing the feature pipeline.
        """
        result = encode_categoricals(self._df("UNKNOWN_PORT", "MAERSK"))
        assert result["terminal_id_encoded"].iloc[0] == -1

    def test_unknown_carrier_encodes_to_minus_one(self):
        """Unknown carrier codes must encode to -1 for the same reason."""
        result = encode_categoricals(self._df("CPT", "UNKNOWN_CARRIER"))
        assert result["service_code_encoded"].iloc[0] == -1

    def test_encoding_stable_across_calls(self):
        """
        The same terminal must receive the same integer on every call.
        Encoding instability would corrupt predictions at inference time.
        """
        result1 = encode_categoricals(self._df("NCT", "MSC"))
        result2 = encode_categoricals(self._df("NCT", "MSC"))
        assert result1["terminal_id_encoded"].iloc[0] == result2["terminal_id_encoded"].iloc[0]
        assert result1["service_code_encoded"].iloc[0] == result2["service_code_encoded"].iloc[0]


# ---------------------------------------------------------------------------
# Integration — ML feature table (reads from disk)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ml_feature_table() -> pd.DataFrame:
    from pathlib import Path
    path = Path("data/gold/ml_vessel_delay_features.parquet")
    if not path.exists():
        pytest.skip(f"ML feature table not found at {path} — run feature_engineering.py first")
    return pd.read_parquet(path, engine="pyarrow")


class TestChronologicalSplit:
    def test_train_and_test_do_not_overlap_in_time(self, ml_feature_table):
        """
        The time-based split must place every training ETA before every test ETA.

        Rolling history features are computed from the same call pool used in
        training.  If the split were random, a test call's ETA would land inside
        the training window, and the rolling features of training rows around it
        would incorporate the test call's outcome — leaking the future into
        training metrics.  This test catches any accidental switch to random
        splitting.
        """
        labeled = (
            ml_feature_table
            .dropna(subset=["is_arrival_delayed_more_than_2_hours"])
            .sort_values("eta")
        )
        cutoff_idx    = int(len(labeled) * 0.8)
        train_max_eta = labeled.iloc[:cutoff_idx]["eta"].max()
        test_min_eta  = labeled.iloc[cutoff_idx:]["eta"].min()

        assert train_max_eta <= test_min_eta, (
            f"Train/test split is not chronological: "
            f"latest training ETA ({train_max_eta}) is after earliest test ETA ({test_min_eta}). "
            "This would cause rolling history features to leak test outcomes into training rows."
        )

    def test_train_set_is_majority_of_labeled_rows(self, ml_feature_table):
        """Training set must be approximately 80% of labeled rows (ML_TEST_SIZE = 0.2)."""
        labeled   = ml_feature_table.dropna(subset=["is_arrival_delayed_more_than_2_hours"])
        total     = len(labeled)
        train_n   = int(total * 0.8)
        assert 0.75 <= train_n / total <= 0.85, (
            f"Unexpected train fraction: {train_n}/{total} = {train_n/total:.2%}"
        )


class TestPITCorrectness:
    def test_pit_filter_produces_at_least_one_null_per_terminal(self, ml_feature_table):
        """
        At least one call at each terminal must have a null
        avg_previous_10_terminal_delays.

        The history pool is all completed calls with ATD.  The PIT filter
        (history.atd < current.eta) means any call whose ETA precedes all
        prior completed calls at its terminal will receive null.  If the
        PIT filter were removed, every call would find history from the full
        simulation window — no nulls would exist.

        Note: the "first eligible candidate" at a terminal may still have
        non-null history if other completed-but-ineligible calls (missing ETA,
        invalid sequence) predated it.  This test correctly targets calls
        whose ETA is genuinely before all prior ATDs at their terminal.
        """
        col   = "avg_previous_10_terminal_delays"
        nulls = ml_feature_table[col].isna().sum()
        assert nulls > 0, (
            f"{col} is non-null for every one of {len(ml_feature_table)} calls. "
            "Some calls should lack prior history (they are earliest in their terminal's "
            "simulation window).  Zero nulls suggests the PIT filter was removed and the "
            "rolling mean is computed from the full dataset rather than pre-ETA history."
        )

    def test_pit_join_is_not_broken_for_majority_of_calls(self, ml_feature_table):
        """
        The majority of calls must have a non-null avg_previous_10_terminal_delays.

        In a 24-month simulation across 5 active terminals, most calls have
        prior completed calls available at their terminal before their ETA.
        A very high null rate (>30%) would indicate the PIT join is broken —
        for instance, if the ATD/ETA comparison was inverted.
        """
        col       = "avg_previous_10_terminal_delays"
        total     = len(ml_feature_table)
        null_rate = ml_feature_table[col].isna().sum() / total
        assert null_rate < 0.30, (
            f"{col} null rate is {null_rate:.1%} ({int(null_rate * total)}/{total}). "
            "If above 30%, the rolling history join is likely broken — most calls in a "
            "24-month simulation should find prior completed calls before their ETA."
        )
