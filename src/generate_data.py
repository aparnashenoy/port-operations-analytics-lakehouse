"""
Synthetic data generator for the port operations analytics lakehouse.

Generates realistic, messy operational data across five terminals covering
one simulation year. Controlled data quality issues are injected into every
table so downstream bronze-layer validation and quality-gate tests have
realistic failure cases to catch.

Run:
    python src/generate_data.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    DIRTY_DATA_RATE,
    RAW_DIR,
    SIMULATION_END,
    SIMULATION_START,
    SYNTHETIC_SEED,
    ensure_dirs,
)

fake = Faker()
Faker.seed(SYNTHETIC_SEED)
rng = np.random.default_rng(SYNTHETIC_SEED)
random.seed(SYNTHETIC_SEED)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

VESSEL_CLASSES = ["Feeder", "Sub-Panamax", "Panamax", "Post-Panamax", "ULCV"]

VESSEL_CLASS_TEU: dict[str, tuple[int, int]] = {
    "Feeder":        (300,   999),
    "Sub-Panamax":   (1000,  2999),
    "Panamax":       (3000,  4999),
    "Post-Panamax":  (5000,  14999),
    "ULCV":          (15000, 24000),
}

# Planned turnaround window (hours) by vessel class
VESSEL_CLASS_TURNAROUND: dict[str, tuple[float, float]] = {
    "Feeder":        (8,  18),
    "Sub-Panamax":   (14, 28),
    "Panamax":       (20, 40),
    "Post-Panamax":  (28, 60),
    "ULCV":          (40, 90),
}

CARRIERS = [
    "MAERSK", "MSC", "CMACGM", "EVERGREEN", "COSCO",
    "ONE", "HAPAG", "APL", "ZIM", "PIL",
]

DELAY_REASONS = ["weather", "equipment", "documentation", "congestion", "none"]
DELAY_REASON_BASE_WEIGHTS = [0.15, 0.25, 0.15, 0.20, 0.25]

CONTAINER_SIZES = ["20ft", "40ft"]
CONTAINER_TYPES = ["dry", "reefer", "hazmat", "oog"]
CONTAINER_TYPE_WEIGHTS = [0.75, 0.15, 0.07, 0.03]

MOVE_TYPES = ["discharge", "load", "shift"]
MOVE_TYPE_WEIGHTS = [0.45, 0.45, 0.10]

FLAG_STATES = ["PA", "LR", "MH", "BS", "CY", "MT", "BH", "AG", "TC", "KN"]

# ---------------------------------------------------------------------------
# Terminal catalogue
# ---------------------------------------------------------------------------

TERMINALS: list[dict] = [
    {
        "terminal_id":            "NCT",
        "terminal_name":          "Northgate Container Terminal",
        "berth_count":            6,
        "crane_count":            12,
        "max_vessel_class":       "ULCV",
        "base_productivity_mph":  28,   # moves per crane per hour (design rate)
        "delay_bias":             0.10, # additive probability on top of baseline
        "latitude":               51.5074,
        "longitude":              0.1278,
    },
    {
        "terminal_id":            "EFT",
        "terminal_name":          "Eastport Freight Terminal",
        "berth_count":            4,
        "crane_count":            8,
        "max_vessel_class":       "Post-Panamax",
        "base_productivity_mph":  24,
        "delay_bias":             0.05,
        "latitude":               51.4800,
        "longitude":              0.2100,
    },
    {
        "terminal_id":            "SLH",
        "terminal_name":          "Southside Logistics Hub",
        "berth_count":            3,
        "crane_count":            5,
        "max_vessel_class":       "Panamax",
        "base_productivity_mph":  20,
        "delay_bias":             0.15,
        "latitude":               51.4600,
        "longitude":              0.0900,
    },
    {
        "terminal_id":            "WIT",
        "terminal_name":          "Westquay Industrial Terminal",
        "berth_count":            3,
        "crane_count":            4,
        "max_vessel_class":       "Sub-Panamax",
        "base_productivity_mph":  18,
        "delay_bias":             0.20,
        "latitude":               51.4900,
        "longitude":              0.0500,
    },
    {
        "terminal_id":            "CPT",
        "terminal_name":          "Central Port Terminal",
        "berth_count":            4,
        "crane_count":            7,
        "max_vessel_class":       "Panamax",
        "base_productivity_mph":  22,
        "delay_bias":             0.08,
        "latitude":               51.5000,
        "longitude":              0.1100,
    },
]

TERMINAL_IDS = [t["terminal_id"] for t in TERMINALS]
TERMINAL_MAP: dict[str, dict] = {t["terminal_id"]: t for t in TERMINALS}

# Traffic distribution across terminals
TERMINAL_TRAFFIC_WEIGHTS = [0.35, 0.25, 0.15, 0.10, 0.15]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_terminal_metadata() -> pd.DataFrame:
    """Return the static terminal reference table; no randomness applied."""
    return pd.DataFrame(TERMINALS)


def generate_vessels(n: int = 100) -> pd.DataFrame:
    """
    Generate a vessel registry with IMO-style identifiers, vessel classes,
    carriers, and capacity figures.
    """
    vessel_class_weights = [0.20, 0.25, 0.25, 0.20, 0.10]
    records = []
    for i in range(n):
        vclass = str(rng.choice(VESSEL_CLASSES, p=vessel_class_weights))
        teu_lo, teu_hi = VESSEL_CLASS_TEU[vclass]
        capacity_teu = int(rng.integers(teu_lo, teu_hi + 1))
        records.append(
            {
                "vessel_id":      f"IMO{9000000 + i:07d}",
                "vessel_name":    f"MV {fake.last_name().upper()} {fake.word().upper()}",
                "vessel_class":   vclass,
                "carrier_code":   str(rng.choice(CARRIERS)),
                "flag_state":     str(rng.choice(FLAG_STATES)),
                "capacity_teu":   capacity_teu,
                "year_built":     int(rng.integers(1995, 2024)),
                "gross_tonnage":  int(capacity_teu * rng.uniform(1.8, 2.4)),
            }
        )
    return pd.DataFrame(records)


def generate_weather(start: str, end: str) -> pd.DataFrame:
    """
    Generate one row per (date, terminal) for every calendar day in the
    simulation window.  Wind speed and sea state follow a seasonal cosine
    curve that peaks in January with occasional storm spikes.
    """
    dates = pd.date_range(start=start, end=end, freq="D")
    records = []

    for date in dates:
        month = date.month
        # winter_factor peaks near 1.5 in Jan, troughs near 0.5 in Jul
        winter_factor = 1.0 + 0.5 * float(np.cos(np.pi * (month - 7) / 6))

        for terminal_id in TERMINAL_IDS:
            base_wind = float(rng.uniform(5, 15)) * winter_factor
            if rng.random() < 0.05:           # storm spike on ~5 % of days
                base_wind *= float(rng.uniform(2.0, 3.5))
            base_wind = min(base_wind, 65.0)

            visibility = max(0.5, float(rng.uniform(3, 12)) / winter_factor)
            wave_height = max(0.1, base_wind * float(rng.uniform(0.05, 0.12)))
            precipitation = max(0.0, float(rng.normal(2, 3)) * winter_factor)

            severity = min(
                1.0,
                (base_wind / 65) * 0.45
                + (1.0 - min(visibility, 10.0) / 10.0) * 0.30
                + (min(wave_height, 6.0) / 6.0) * 0.25,
            )
            impact_factor = 1.0 + severity * float(rng.uniform(0.5, 1.5))

            records.append(
                {
                    "weather_date":         date.date(),
                    "terminal_id":          terminal_id,
                    "wind_speed_knots":     round(base_wind, 1),
                    "wave_height_m":        round(wave_height, 2),
                    "visibility_nm":        round(visibility, 1),
                    "precipitation_mm":     round(precipitation, 1),
                    "severity_index":       round(severity, 4),
                    "weather_impact_factor": round(impact_factor, 4),
                }
            )

    return pd.DataFrame(records)


def _sample_delay(terminal: dict, weather_impact: float) -> tuple[float, str]:
    """
    Return (delay_hours, delay_reason).

    Delay probability is the baseline 30 % plus the terminal's configured
    bias, further boosted when weather is severe.  Reason weights shift
    toward "weather" when the impact factor is high.
    """
    delay_prob = 0.30 + terminal["delay_bias"]
    if weather_impact > 1.3:
        delay_prob = min(0.90, delay_prob + (weather_impact - 1.0) * 0.4)

    if rng.random() > delay_prob:
        return 0.0, "none"

    weights = list(DELAY_REASON_BASE_WEIGHTS)
    if weather_impact > 1.3:
        weights[0] = min(0.55, weights[0] + (weather_impact - 1.0) * 0.3)
        total = sum(weights)
        weights = [w / total for w in weights]

    reason = str(rng.choice(DELAY_REASONS, p=weights))
    delay_hours = float(rng.exponential(scale=4.0)) + 0.5
    if reason == "weather":
        delay_hours *= weather_impact

    return round(delay_hours, 2), reason


def generate_vessel_calls(
    vessels: pd.DataFrame,
    weather: pd.DataFrame,
    n_calls: int = 2000,
) -> pd.DataFrame:
    """
    Generate vessel call records with planned/actual arrival and departure
    timestamps, delay attribution, and cargo volumes.

    Data quality issues injected:
      - Missing ETA (~1.5 % of rows)
      - Missing ATA/ATD, status set to 'in_progress' (~1 %)
      - Invalid timestamp sequences: ATA > ATD (~0.5 %)
      - Duplicate records with a later record_created_at (~1 %)
      - Late record updates: record_created_at far after ATD (~0.8 %)
    """
    dates = pd.date_range(start=SIMULATION_START, end=SIMULATION_END, freq="D")
    dates_np = dates.to_numpy()  # numpy datetime64 array for rng.choice

    weather_impact_lookup: dict[tuple, float] = (
        weather.set_index(["weather_date", "terminal_id"])["weather_impact_factor"]
        .to_dict()
    )

    records = []
    for call_idx in range(n_calls):
        terminal_id = str(rng.choice(TERMINAL_IDS, p=TERMINAL_TRAFFIC_WEIGHTS))
        terminal = TERMINAL_MAP[terminal_id]

        # Restrict to vessels the terminal can physically berth
        max_cls_idx = VESSEL_CLASSES.index(terminal["max_vessel_class"])
        compatible = vessels[
            vessels["vessel_class"].map(VESSEL_CLASSES.index) <= max_cls_idx
        ]
        vessel = compatible.iloc[int(rng.integers(0, len(compatible)))]

        # Pick a random arrival date/time within the simulation window
        eta_date = pd.Timestamp(dates_np[int(rng.integers(0, len(dates_np)))])
        eta = eta_date + pd.Timedelta(
            hours=int(rng.integers(0, 24)),
            minutes=int(rng.choice([0, 15, 30, 45])),
        )

        weather_key = (eta.date(), terminal_id)
        weather_impact = weather_impact_lookup.get(weather_key, 1.0)

        # ATA: actual arrival, typically slightly after ETA
        arrival_offset_h = float(rng.normal(loc=0.5, scale=2.5))
        ata = eta + pd.Timedelta(hours=arrival_offset_h)

        lo, hi = VESSEL_CLASS_TURNAROUND[str(vessel["vessel_class"])]
        planned_hours = float(rng.uniform(lo, hi))
        etd = eta + pd.Timedelta(hours=planned_hours)

        delay_hours, delay_reason = _sample_delay(terminal, weather_impact)
        atd = ata + pd.Timedelta(hours=planned_hours + delay_hours)

        capacity = int(vessel["capacity_teu"])
        fill_rate = float(rng.uniform(0.55, 0.98))
        planned_teu = int(capacity * fill_rate)
        actual_teu = int(planned_teu * float(rng.uniform(0.92, 1.05)))

        berth_num = int(rng.integers(1, terminal["berth_count"] + 1))

        records.append(
            {
                "vessel_call_id":           f"VC{call_idx + 1:06d}",
                "vessel_id":                vessel["vessel_id"],
                "vessel_name":              vessel["vessel_name"],
                "vessel_class":             vessel["vessel_class"],
                "carrier_code":             vessel["carrier_code"],
                "terminal_id":              terminal_id,
                "berth_id":                 f"{terminal_id}-B{berth_num:02d}",
                "eta":                      eta,
                "ata":                      ata,
                "etd":                      etd,
                "atd":                      atd,
                "planned_turnaround_hours": round(planned_hours, 2),
                "actual_turnaround_hours":  round((atd - ata).total_seconds() / 3600, 2),
                "delay_hours":              round(delay_hours, 2),
                "delay_reason":             delay_reason,
                "planned_cargo_teu":        planned_teu,
                "actual_cargo_teu":         actual_teu,
                "weather_impact_factor":    round(float(weather_impact), 4),
                "status":                   "completed",
                "record_created_at":        ata + pd.Timedelta(hours=float(rng.uniform(0, 2))),
            }
        )

    df = pd.DataFrame(records)
    n = len(df)

    # -- Data quality injections --

    # 1. Missing ETA
    idx = rng.choice(df.index.to_numpy(), size=max(1, int(n * 0.015)), replace=False)
    df.loc[idx, "eta"] = pd.NaT

    # 2. Missing ATA + ATD → vessel not yet departed
    idx = rng.choice(df.index.to_numpy(), size=max(1, int(n * 0.010)), replace=False)
    df.loc[idx, ["ata", "atd"]] = pd.NaT
    df.loc[idx, "status"] = "in_progress"

    # 3. Invalid sequence: ATA after ATD
    idx = rng.choice(df.index.to_numpy(), size=max(1, int(n * 0.005)), replace=False)
    df.loc[idx, "ata"] = df.loc[idx, "atd"] + pd.Timedelta(hours=1)

    # 4. Duplicate records (same call, newer created_at timestamp)
    dup_idx = rng.choice(df.index.to_numpy(), size=max(1, int(n * 0.010)), replace=False)
    dupes = df.loc[dup_idx].copy()
    dupes["record_created_at"] = dupes["record_created_at"] + pd.Timedelta(hours=float(rng.uniform(1, 48)))
    df = pd.concat([df, dupes], ignore_index=True)

    # 5. Late record updates
    atd_known = df.dropna(subset=["atd"]).index.to_numpy()
    late_idx = rng.choice(atd_known, size=max(1, int(len(atd_known) * 0.008)), replace=False)
    df.loc[late_idx, "record_created_at"] = df.loc[late_idx, "atd"] + pd.to_timedelta(
        rng.uniform(24, 72, size=len(late_idx)), unit="h"
    )

    return df


def generate_crane_assignments(
    vessel_calls: pd.DataFrame,
    n_target: int = 10_000,
) -> pd.DataFrame:
    """
    Generate crane assignment records linked to vessel calls.

    Each eligible call receives 1–N crane assignments proportioned so the
    total approaches n_target.  Assignments include planned and actual
    start/end times and per-crane productivity figures that vary by terminal.

    Data quality issues injected:
      - Crane overlap conflicts: same crane, overlapping actual times (~2 %)
      - Missing actual start/end (~1.5 %)
      - Actual end before actual start (~0.5 %)
    """
    eligible = vessel_calls[vessel_calls["status"].isin(["completed", "in_progress"])].dropna(
        subset=["ata"]
    )

    avg_per_call = max(1, n_target // max(1, len(eligible)))

    records = []
    assignment_counter = 0

    for _, call in eligible.iterrows():
        terminal = TERMINAL_MAP[call["terminal_id"]]
        n_cranes = int(rng.integers(1, min(avg_per_call + 3, terminal["crane_count"] + 1)))
        all_cranes = [f"{call['terminal_id']}-CR{i + 1:02d}" for i in range(terminal["crane_count"])]

        call_start = call["ata"]
        if pd.notna(call["atd"]):
            call_end = call["atd"]
        else:
            call_end = call_start + pd.Timedelta(hours=24)

        window_h = max(0.5, (call_end - call_start).total_seconds() / 3600)

        for _ in range(n_cranes):
            assignment_counter += 1
            crane_id = str(rng.choice(all_cranes))

            offset_h = float(rng.uniform(0, window_h * 0.5))
            planned_start = call_start + pd.Timedelta(hours=offset_h)
            planned_h_hi = max(2.1, min(12.0, window_h - offset_h + 1.0))
            planned_h = float(rng.uniform(2.0, planned_h_hi))
            planned_end = planned_start + pd.Timedelta(hours=planned_h)

            actual_start = planned_start + pd.Timedelta(hours=float(rng.normal(0.2, 0.5)))
            actual_h = planned_h * float(rng.uniform(0.85, 1.20))
            actual_end = actual_start + pd.Timedelta(hours=actual_h)

            productivity = terminal["base_productivity_mph"] * float(rng.uniform(0.80, 1.15))

            records.append(
                {
                    "assignment_id":               f"CA{assignment_counter:07d}",
                    "vessel_call_id":               call["vessel_call_id"],
                    "terminal_id":                  call["terminal_id"],
                    "crane_id":                     crane_id,
                    "planned_start":                planned_start,
                    "planned_end":                  planned_end,
                    "actual_start":                 actual_start,
                    "actual_end":                   actual_end,
                    "planned_hours":                round(planned_h, 2),
                    "actual_hours":                 round(actual_h, 2),
                    "productivity_moves_per_hour":  round(productivity, 1),
                    "status":                       "completed",
                }
            )

    df = pd.DataFrame(records)
    n = len(df)
    idx_arr = df.index.to_numpy()

    # 1. Crane overlap: shift actual_start back so it overlaps a prior assignment
    idx = rng.choice(idx_arr, size=max(1, int(n * 0.02)), replace=False)
    df.loc[idx, "actual_start"] = df.loc[idx, "actual_start"] - pd.Timedelta(hours=2)

    # 2. Missing actual times
    idx = rng.choice(idx_arr, size=max(1, int(n * 0.015)), replace=False)
    df.loc[idx, ["actual_start", "actual_end"]] = pd.NaT

    # 3. Actual end before actual start
    has_actual = df.dropna(subset=["actual_start", "actual_end"]).index.to_numpy()
    idx = rng.choice(has_actual, size=max(1, int(n * 0.005)), replace=False)
    df.loc[idx, "actual_end"] = df.loc[idx, "actual_start"] - pd.Timedelta(hours=1)

    return df


def generate_container_moves(
    crane_assignments: pd.DataFrame,
    n_target: int = 100_000,
) -> pd.DataFrame:
    """
    Generate individual container move records linked to crane assignments.

    Move counts per assignment are randomised around the target average.
    Planned vs actual move counts diverge to simulate scope changes during
    operations.

    Data quality issues injected:
      - Missing actual_move_time and actual_duration on completed moves (~2 %)
      - Negative actual_duration_minutes (~0.3 %)
    """
    eligible = crane_assignments.dropna(subset=["actual_start"]).copy()
    avg_per_assignment = max(1, n_target // max(1, len(eligible)))

    records = []
    move_counter = 0

    for _, assignment in eligible.iterrows():
        n_moves = int(rng.integers(
            max(1, avg_per_assignment - 5),
            avg_per_assignment + 6,
        ))

        # Actual scope can differ from planned
        actual_scope = int(n_moves * float(rng.uniform(0.90, 1.08)))

        start = assignment["actual_start"]
        if pd.notna(assignment["actual_end"]):
            window_s = (assignment["actual_end"] - start).total_seconds()
        else:
            window_s = assignment["actual_hours"] * 3600
        window_s = max(window_s, 60.0)

        for j in range(n_moves):
            move_counter += 1
            move_type = str(rng.choice(MOVE_TYPES, p=MOVE_TYPE_WEIGHTS))
            container_size = str(rng.choice(CONTAINER_SIZES, p=[0.40, 0.60]))
            container_type = str(rng.choice(CONTAINER_TYPES, p=CONTAINER_TYPE_WEIGHTS))

            planned_offset_s = j * (window_s / n_moves)
            planned_move_time = start + pd.Timedelta(seconds=planned_offset_s)

            actual_offset_s = max(0.0, planned_offset_s + float(rng.normal(0, 120)))
            actual_move_time = start + pd.Timedelta(seconds=actual_offset_s)

            planned_dur = float(rng.uniform(2.0, 6.0))
            if container_type == "hazmat":
                planned_dur *= 1.5
            elif container_type == "oog":
                planned_dur *= 2.0
            actual_dur = planned_dur * float(rng.uniform(0.80, 1.30))

            # Moves beyond actual_scope are cancelled
            if j >= actual_scope:
                move_status = "cancelled"
                actual_move_time = pd.NaT
                actual_dur_val = None
            elif rng.random() < 0.02:
                move_status = "deferred"
                actual_dur_val = round(actual_dur, 2)
            else:
                move_status = "completed"
                actual_dur_val = round(actual_dur, 2)

            records.append(
                {
                    "move_id":                  f"MV{move_counter:08d}",
                    "assignment_id":            assignment["assignment_id"],
                    "vessel_call_id":           assignment["vessel_call_id"],
                    "terminal_id":              assignment["terminal_id"],
                    "crane_id":                 assignment["crane_id"],
                    "move_type":                move_type,
                    "container_size":           container_size,
                    "container_type":           container_type,
                    "planned_move_time":        planned_move_time,
                    "actual_move_time":         actual_move_time,
                    "planned_duration_minutes": round(planned_dur, 2),
                    "actual_duration_minutes":  actual_dur_val,
                    "move_status":              move_status,
                }
            )

    df = pd.DataFrame(records)
    n = len(df)

    # 1. Nullify actual times on ~2 % of completed moves
    completed_idx = df[df["move_status"] == "completed"].index.to_numpy()
    if len(completed_idx) > 0:
        null_idx = rng.choice(
            completed_idx,
            size=max(1, int(len(completed_idx) * 0.02)),
            replace=False,
        )
        df.loc[null_idx, ["actual_move_time", "actual_duration_minutes"]] = [pd.NaT, None]

    # 2. Negative actual_duration (~0.3 %)
    has_dur = df["actual_duration_minutes"].notna()
    neg_pool = df[has_dur].index.to_numpy()
    if len(neg_pool) > 0:
        neg_idx = rng.choice(neg_pool, size=max(1, int(n * 0.003)), replace=False)
        df.loc[neg_idx, "actual_duration_minutes"] = -df.loc[neg_idx, "planned_duration_minutes"]

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_dirs()
    print("Generating synthetic port operations data ...")
    print(f"  Simulation window : {SIMULATION_START}  →  {SIMULATION_END}")
    print(f"  Random seed       : {SYNTHETIC_SEED}")
    print()

    print("  [1/5] Terminal metadata ...")
    terminals_df = generate_terminal_metadata()
    terminals_df.to_csv(RAW_DIR / "terminal_metadata.csv", index=False)
    print(f"         {len(terminals_df)} terminals  →  terminal_metadata.csv")

    print("  [2/5] Daily weather data ...")
    weather_df = generate_weather(SIMULATION_START, SIMULATION_END)
    weather_df.to_csv(RAW_DIR / "weather_daily.csv", index=False)
    print(f"         {len(weather_df):>8,} rows  →  weather_daily.csv")

    print("  [3/5] Vessel calls ...")
    vessels_df = generate_vessels(n=100)
    calls_df = generate_vessel_calls(vessels_df, weather_df, n_calls=2000)
    calls_df.to_csv(RAW_DIR / "vessel_calls.csv", index=False)
    print(f"         {len(calls_df):>8,} rows  →  vessel_calls.csv  (incl. injected duplicates)")

    print("  [4/5] Crane assignments ...")
    crane_df = generate_crane_assignments(calls_df, n_target=10_000)
    crane_df.to_csv(RAW_DIR / "crane_assignments.csv", index=False)
    print(f"         {len(crane_df):>8,} rows  →  crane_assignments.csv")

    print("  [5/5] Container moves ...")
    moves_df = generate_container_moves(crane_df, n_target=100_000)
    moves_df.to_csv(RAW_DIR / "container_moves.csv", index=False)
    print(f"         {len(moves_df):>8,} rows  →  container_moves.csv")

    print()
    print(f"Done.  Raw files written to: {RAW_DIR}")


if __name__ == "__main__":
    main()
