"""
Unit tests for src/silver_transformations.py.

Each test exercises a single transformation function with a hand-crafted
in-memory DataFrame rather than reading from disk.  This keeps tests fast
and independent of whether the pipeline has been run.

Run: pytest tests/test_transformations.py
"""

import numpy as np
import pandas as pd
import pytest

from src.silver_transformations import (
    add_crane_flags,
    add_vessel_call_flags,
    clean_container_moves,
    compute_delay_minutes,
    deduplicate_vessel_calls,
)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------


def _vc(**overrides) -> pd.DataFrame:
    """
    Return a single-row vessel_calls DataFrame with sensible defaults.
    Pass keyword arguments as lists to override any column(s).
    """
    defaults = {
        "vessel_call_id":  ["VC001"],
        "eta":             [pd.Timestamp("2024-06-01 08:00")],
        "ata":             [pd.Timestamp("2024-06-01 10:00")],
        "etd":             [pd.Timestamp("2024-06-02 08:00")],
        "atd":             [pd.Timestamp("2024-06-02 10:00")],
        "delay_hours":     [2.0],
        "record_created_at": [pd.Timestamp("2024-06-01 06:00")],
        "_ingestion_ts":   [pd.Timestamp("2024-06-01 07:00", tz="UTC")],
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


def _ca(**overrides) -> pd.DataFrame:
    """
    Return a single-row crane_assignments DataFrame with sensible defaults.
    Pass keyword arguments as lists to override any column(s).
    """
    defaults = {
        "assignment_id":  ["A001"],
        "crane_id":       ["CR01"],
        "vessel_call_id": ["VC001"],
        "actual_start":   [pd.Timestamp("2024-06-01 08:00")],
        "actual_end":     [pd.Timestamp("2024-06-01 16:00")],
        "planned_start":  [pd.Timestamp("2024-06-01 08:00")],
        "planned_end":    [pd.Timestamp("2024-06-01 16:00")],
        "planned_hours":  [8.0],
        "actual_hours":   [8.0],
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


# ---------------------------------------------------------------------------
# compute_delay_minutes
# ---------------------------------------------------------------------------


class TestComputeDelayMinutes:
    def test_arrival_delay_computed_when_eta_and_ata_present(self):
        """
        arrival_delay_minutes must equal (ATA - ETA) in minutes.
        This is the primary metric the model predicts and must be exact.
        """
        df = _vc(
            eta=[pd.Timestamp("2024-06-01 08:00")],
            ata=[pd.Timestamp("2024-06-01 09:30")],
        )
        result = compute_delay_minutes(df)
        assert result["arrival_delay_minutes"].iloc[0] == pytest.approx(90.0)

    def test_arrival_delay_null_when_eta_missing(self):
        """Without an ETA there is no reference point — delay must be NaN."""
        # pd.NaT keeps the column as datetime64[ns]; plain None produces object dtype,
        # which causes a TypeError when compute_delay_minutes subtracts columns.
        df = _vc(eta=[pd.NaT])
        result = compute_delay_minutes(df)
        assert pd.isna(result["arrival_delay_minutes"].iloc[0])

    def test_arrival_delay_null_when_ata_missing(self):
        """Without an ATA the vessel has not arrived — delay must be NaN."""
        df = _vc(ata=[pd.NaT])
        result = compute_delay_minutes(df)
        assert pd.isna(result["arrival_delay_minutes"].iloc[0])

    def test_arrival_delay_negative_for_early_arrival(self):
        """A vessel arriving before its ETA produces a negative delay (early)."""
        df = _vc(
            eta=[pd.Timestamp("2024-06-01 10:00")],
            ata=[pd.Timestamp("2024-06-01 09:30")],
        )
        result = compute_delay_minutes(df)
        assert result["arrival_delay_minutes"].iloc[0] == pytest.approx(-30.0)

    def test_departure_delay_computed_when_etd_and_atd_present(self):
        """departure_delay_minutes must equal (ATD - ETD) in minutes."""
        df = _vc(
            etd=[pd.Timestamp("2024-06-02 08:00")],
            atd=[pd.Timestamp("2024-06-02 10:00")],
        )
        result = compute_delay_minutes(df)
        assert result["departure_delay_minutes"].iloc[0] == pytest.approx(120.0)

    def test_departure_delay_null_when_atd_missing(self):
        """Without ATD the vessel has not departed — departure_delay_minutes must be NaN."""
        df = _vc(atd=[pd.NaT])
        result = compute_delay_minutes(df)
        assert pd.isna(result["departure_delay_minutes"].iloc[0])

    def test_on_time_arrival_produces_zero_delay(self):
        """ATA == ETA → arrival_delay_minutes == 0."""
        ts = pd.Timestamp("2024-06-01 09:00")
        df = _vc(eta=[ts], ata=[ts])
        result = compute_delay_minutes(df)
        assert result["arrival_delay_minutes"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# deduplicate_vessel_calls
# ---------------------------------------------------------------------------


class TestDeduplicateVesselCalls:
    def _two_rows(self, delay_hours=(1.0, 3.0), created_at_offsets=(0, 6)):
        """Helper: return two rows for VC001 with different record_created_at times."""
        # Use UTC-aware base so _ingestion_ts is tz-aware (required by tz_convert)
        base = pd.Timestamp("2024-06-01 06:00", tz="UTC")
        return _vc(
            vessel_call_id=["VC001", "VC001"],
            delay_hours=list(delay_hours),
            record_created_at=[
                base + pd.Timedelta(hours=created_at_offsets[0]),
                base + pd.Timedelta(hours=created_at_offsets[1]),
            ],
            _ingestion_ts=[
                base + pd.Timedelta(hours=created_at_offsets[0], minutes=5),
                base + pd.Timedelta(hours=created_at_offsets[1], minutes=5),
            ],
            eta=2 * [pd.Timestamp("2024-06-01 08:00")],
            ata=2 * [pd.Timestamp("2024-06-01 10:00")],
            etd=2 * [pd.Timestamp("2024-06-02 08:00")],
            atd=2 * [pd.Timestamp("2024-06-02 10:00")],
        )

    def test_single_row_per_vessel_call_after_dedup(self):
        """No vessel_call_id may appear more than once after deduplication."""
        df = self._two_rows()
        result, _ = deduplicate_vessel_calls(df)
        assert result["vessel_call_id"].duplicated().sum() == 0

    def test_most_recent_record_retained(self):
        """
        When two records exist for the same call, the one with the later
        record_created_at is kept.  delay_hours serves as a row identity marker.
        """
        df = self._two_rows(delay_hours=(1.0, 9.0), created_at_offsets=(0, 6))
        result, _ = deduplicate_vessel_calls(df)
        # The row created 6 hours later (delay_hours=9.0) must survive
        assert result["delay_hours"].iloc[0] == pytest.approx(9.0)

    def test_returns_correct_drop_count(self):
        """deduplicate_vessel_calls must return the number of rows removed."""
        # 3 rows: 2 for VC001 (one dup) + 1 for VC002
        df = _vc(
            vessel_call_id=["VC001", "VC001", "VC002"],
            record_created_at=[
                pd.Timestamp("2024-06-01 06:00"),
                pd.Timestamp("2024-06-01 07:00"),
                pd.Timestamp("2024-06-01 08:00"),
            ],
            _ingestion_ts=[
                pd.Timestamp("2024-06-01 06:05", tz="UTC"),
                pd.Timestamp("2024-06-01 07:05", tz="UTC"),
                pd.Timestamp("2024-06-01 08:05", tz="UTC"),
            ],
            delay_hours=[1.0, 2.0, 3.0],
            eta=3 * [pd.Timestamp("2024-06-01 08:00")],
            ata=3 * [pd.Timestamp("2024-06-01 10:00")],
            etd=3 * [pd.Timestamp("2024-06-02 08:00")],
            atd=3 * [pd.Timestamp("2024-06-02 10:00")],
        )
        _, n_dropped = deduplicate_vessel_calls(df)
        assert n_dropped == 1

    def test_unique_rows_unchanged(self):
        """A DataFrame with no duplicates must return the same row count."""
        df = _vc()  # single row, no dups
        result, n_dropped = deduplicate_vessel_calls(df)
        assert len(result) == 1
        assert n_dropped == 0


# ---------------------------------------------------------------------------
# add_vessel_call_flags
# ---------------------------------------------------------------------------


class TestVesselCallFlags:
    def test_missing_eta_flag_set_when_eta_null(self):
        """missing_eta_flag must be True when eta is NULL."""
        df = _vc(eta=[None])
        result = add_vessel_call_flags(df)
        assert result["missing_eta_flag"].iloc[0]

    def test_missing_eta_flag_clear_when_eta_present(self):
        """missing_eta_flag must be False when eta is not NULL."""
        df = _vc()
        result = add_vessel_call_flags(df)
        assert not result["missing_eta_flag"].iloc[0]

    def test_missing_ata_flag_set_when_ata_null(self):
        """missing_ata_flag must be True when ata is NULL."""
        df = _vc(ata=[None])
        result = add_vessel_call_flags(df)
        assert result["missing_ata_flag"].iloc[0]

    def test_invalid_arrival_sequence_flag_when_ata_after_atd(self):
        """
        ATA > ATD is physically impossible: a vessel cannot depart before it
        arrives.  This flag must be set to True in that case.
        """
        df = _vc(
            ata=[pd.Timestamp("2024-06-02 15:00")],  # after the ATD below
            atd=[pd.Timestamp("2024-06-02 10:00")],
        )
        result = add_vessel_call_flags(df)
        assert result["invalid_arrival_sequence_flag"].iloc[0]

    def test_invalid_arrival_sequence_flag_clear_for_valid_sequence(self):
        """ATA < ATD (the normal case) must not be flagged."""
        df = _vc()  # default: ATA 10:00, ATD next-day 10:00
        result = add_vessel_call_flags(df)
        assert not result["invalid_arrival_sequence_flag"].iloc[0]

    def test_invalid_departure_sequence_flag_when_etd_before_eta(self):
        """ETD < ETA means planned departure is before planned arrival: a scheduling error."""
        df = _vc(
            eta=[pd.Timestamp("2024-06-01 10:00")],
            etd=[pd.Timestamp("2024-06-01 06:00")],  # before ETA
        )
        result = add_vessel_call_flags(df)
        assert result["invalid_departure_sequence_flag"].iloc[0]

    def test_invalid_departure_sequence_flag_clear_for_valid_schedule(self):
        """Normal ETD > ETA must not produce a departure sequence flag."""
        df = _vc()  # default: ETA Jun-01 08:00, ETD Jun-02 08:00
        result = add_vessel_call_flags(df)
        assert not result["invalid_departure_sequence_flag"].iloc[0]


# ---------------------------------------------------------------------------
# Crane assignment flags
# ---------------------------------------------------------------------------


class TestCraneAssignmentFlags:
    def test_invalid_crane_time_flag_when_end_before_start(self):
        """
        actual_end < actual_start is physically invalid.
        add_crane_flags must set invalid_crane_time_flag = True.
        """
        df = _ca(
            actual_start=[pd.Timestamp("2024-06-01 12:00")],
            actual_end=[pd.Timestamp("2024-06-01 08:00")],   # before start
        )
        result = add_crane_flags(df)
        assert result["invalid_crane_time_flag"].iloc[0]

    def test_valid_crane_record_has_end_after_start(self):
        """A crane assignment with actual_end > actual_start must not be flagged invalid."""
        df = _ca()  # default: 08:00 → 16:00
        result = add_crane_flags(df)
        assert not result["invalid_crane_time_flag"].iloc[0]

    def test_crane_overlap_detected_for_same_crane(self):
        """
        Two overlapping assignments for the same crane must both receive
        crane_overlap_flag = True.  Both the earlier and later party in the
        conflict are flagged so either can be investigated.
        """
        df = pd.DataFrame({
            "assignment_id":  ["A001", "A002"],
            "crane_id":       ["CR01", "CR01"],      # same crane
            "vessel_call_id": ["VC001", "VC002"],
            # A002 starts before A001 ends → overlap
            "actual_start":   [pd.Timestamp("2024-06-01 08:00"),
                               pd.Timestamp("2024-06-01 12:00")],
            "actual_end":     [pd.Timestamp("2024-06-01 16:00"),
                               pd.Timestamp("2024-06-01 20:00")],
            "planned_start":  [pd.Timestamp("2024-06-01 08:00"),
                               pd.Timestamp("2024-06-01 12:00")],
            "planned_end":    [pd.Timestamp("2024-06-01 16:00"),
                               pd.Timestamp("2024-06-01 20:00")],
            "planned_hours":  [8.0, 8.0],
            "actual_hours":   [8.0, 8.0],
        })
        result = add_crane_flags(df)
        assert result["crane_overlap_flag"].all(), (
            "Both parties in a crane conflict must be flagged"
        )

    def test_crane_no_overlap_for_sequential_assignments(self):
        """
        Assignments that touch at exactly one endpoint (back-to-back) are not
        overlapping.  crane_overlap_flag must be False for both rows.
        """
        df = pd.DataFrame({
            "assignment_id":  ["A001", "A002"],
            "crane_id":       ["CR01", "CR01"],
            "vessel_call_id": ["VC001", "VC002"],
            # A001 ends exactly when A002 starts — not an overlap
            "actual_start":   [pd.Timestamp("2024-06-01 08:00"),
                               pd.Timestamp("2024-06-01 16:00")],
            "actual_end":     [pd.Timestamp("2024-06-01 16:00"),
                               pd.Timestamp("2024-06-01 20:00")],
            "planned_start":  [pd.Timestamp("2024-06-01 08:00"),
                               pd.Timestamp("2024-06-01 16:00")],
            "planned_end":    [pd.Timestamp("2024-06-01 16:00"),
                               pd.Timestamp("2024-06-01 20:00")],
            "planned_hours":  [8.0, 4.0],
            "actual_hours":   [8.0, 4.0],
        })
        result = add_crane_flags(df)
        assert not result["crane_overlap_flag"].any(), (
            "Back-to-back assignments that share only an endpoint are not overlaps"
        )

    def test_crane_overlap_does_not_cross_different_cranes(self):
        """
        Two assignments that overlap in time but belong to different cranes
        must NOT be flagged — each crane is an independent resource.
        """
        df = pd.DataFrame({
            "assignment_id":  ["A001", "A002"],
            "crane_id":       ["CR01", "CR02"],   # different cranes
            "vessel_call_id": ["VC001", "VC001"],
            "actual_start":   [pd.Timestamp("2024-06-01 08:00"),
                               pd.Timestamp("2024-06-01 10:00")],
            "actual_end":     [pd.Timestamp("2024-06-01 16:00"),
                               pd.Timestamp("2024-06-01 18:00")],
            "planned_start":  [pd.Timestamp("2024-06-01 08:00"),
                               pd.Timestamp("2024-06-01 10:00")],
            "planned_end":    [pd.Timestamp("2024-06-01 16:00"),
                               pd.Timestamp("2024-06-01 18:00")],
            "planned_hours":  [8.0, 8.0],
            "actual_hours":   [8.0, 8.0],
        })
        result = add_crane_flags(df)
        assert not result["crane_overlap_flag"].any(), (
            "Time overlaps on different cranes must not trigger the overlap flag"
        )


# ---------------------------------------------------------------------------
# clean_container_moves
# ---------------------------------------------------------------------------


class TestCleanContainerMoves:
    def _moves(self, actual: float, planned: float) -> pd.DataFrame:
        return pd.DataFrame({
            "actual_duration_minutes":  [actual],
            "planned_duration_minutes": [planned],
        })

    def test_negative_actual_duration_is_nulled(self):
        """
        Negative actual_duration_minutes is a data entry error.
        clean_container_moves must replace it with NaN rather than keeping
        the invalid value or dropping the row.
        """
        result_df, n_nulled = clean_container_moves(self._moves(-15.0, 20.0))
        assert pd.isna(result_df["actual_duration_minutes"].iloc[0])
        assert n_nulled == 1

    def test_positive_actual_duration_unchanged(self):
        """Valid positive durations must pass through unmodified."""
        result_df, n_nulled = clean_container_moves(self._moves(25.0, 20.0))
        assert result_df["actual_duration_minutes"].iloc[0] == pytest.approx(25.0)
        assert n_nulled == 0

    def test_zero_duration_unchanged(self):
        """Zero is a valid (if unusual) duration and must not be nulled."""
        result_df, n_nulled = clean_container_moves(self._moves(0.0, 20.0))
        assert result_df["actual_duration_minutes"].iloc[0] == pytest.approx(0.0)
        assert n_nulled == 0

    def test_duration_variance_equals_actual_minus_planned(self):
        """duration_variance_minutes = actual_duration - planned_duration."""
        result_df, _ = clean_container_moves(self._moves(25.0, 20.0))
        assert result_df["duration_variance_minutes"].iloc[0] == pytest.approx(5.0)

    def test_duration_variance_null_when_actual_was_negative(self):
        """
        When actual_duration is nulled out, duration_variance must also be NaN.
        A variance of (NaN - planned) is undefined and must not be computed.
        """
        result_df, _ = clean_container_moves(self._moves(-5.0, 20.0))
        assert pd.isna(result_df["duration_variance_minutes"].iloc[0])

    def test_returns_count_of_nulled_rows(self):
        """The returned integer must equal exactly the number of negatives nulled."""
        df = pd.DataFrame({
            "actual_duration_minutes":  [-1.0, 5.0, -3.0, 10.0],
            "planned_duration_minutes": [10.0, 10.0, 10.0, 10.0],
        })
        _, n_nulled = clean_container_moves(df)
        assert n_nulled == 2
