"""Shared paths and constants for the LCC entry analysis.

Every script imports the directory locations and the handful of fixed
parameters from here so the pipeline stays consistent and reproducible.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DB1B_DIR = DATA_DIR / "raw" / "db1b"
RAW_T100_DIR = DATA_DIR / "raw" / "t100"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"

# Treatment definition: a market is treated once any of these carriers appears.
# Sun Country (SY) is excluded; it ran a leisure-charter model for most of the
# window and only moved to an ultra-low-cost model late in 2017.
LCC_CARRIERS = ["WN", "B6", "NK", "F9", "G4"]

# Panel spans 2015 Q1 to 2019 Q4 (20 quarters). 2015 is the common pre-entry
# baseline year; the window stops before the 2020 COVID-19 demand shock.
BASELINE_YEAR = 2015
START_YEAR = 2015

# DB1B fare screen (US dollars). Below $10 is mostly non-revenue tickets;
# above $2,500 is typically premium or miscoded international travel.
FARE_MIN = 10
FARE_MAX = 2500

# Event-study window, in quarters either side of entry.
EVENT_WINDOW = 8

# Seed for every stochastic step (matching, forest, cross-fitting).
SEED = 42


def make_output_dirs() -> None:
    """Create the interim, processed and output directories if missing."""
    for directory in (INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
