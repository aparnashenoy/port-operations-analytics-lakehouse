"""
Central configuration for the port operations analytics lakehouse.

All path resolution and runtime parameters live here so every module
imports from a single source of truth rather than hardcoding paths.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Medallion layer directories
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
RAW_DIR = DATA_DIR / "raw"
QUARANTINE_DIR = DATA_DIR / "quarantine"

# DuckDB catalog (single in-process file for the gold layer)
DUCKDB_PATH = GOLD_DIR / "port_analytics.duckdb"

# ---------------------------------------------------------------------------
# Data generation parameters
# ---------------------------------------------------------------------------

SYNTHETIC_SEED   = int(os.getenv("SYNTHETIC_SEED",   "42"))
SIMULATION_START = os.getenv("SIMULATION_START", "2023-01-01")
SIMULATION_END   = os.getenv("SIMULATION_END",   "2024-12-31")

# ---------------------------------------------------------------------------
# Pipeline behaviour
# ---------------------------------------------------------------------------

# Fraction of raw records that will have injected quality issues (0.0–1.0)
DIRTY_DATA_RATE = float(os.getenv("DIRTY_DATA_RATE", "0.04"))

# Turnaround time bounds used by quality checks (hours)
MIN_TURNAROUND_HOURS = 1
MAX_TURNAROUND_HOURS = 168  # 7 days — beyond this is flagged for review

# A port call is labelled "delayed" when actual turnaround exceeds the
# agreed window by more than this threshold.
DELAY_THRESHOLD_HOURS = float(os.getenv("DELAY_THRESHOLD_HOURS", "2.0"))

# ---------------------------------------------------------------------------
# ML configuration
# ---------------------------------------------------------------------------

ML_TEST_SIZE    = float(os.getenv("ML_TEST_SIZE", "0.2"))
ML_RANDOM_STATE = SYNTHETIC_SEED

MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create all data layer directories if they do not already exist."""
    for directory in (RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, QUARANTINE_DIR, MODELS_DIR, OUTPUTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
