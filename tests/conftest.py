"""
Pytest configuration for the port-operations-analytics-lakehouse test suite.

Adds the project root to sys.path so all test files can import from src.*
without modifying the Python environment.
"""

import sys
from pathlib import Path

# Project root is one level above the tests/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
