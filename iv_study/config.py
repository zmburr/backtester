"""Configuration for the IV top-timing study."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVERSAL_CSV = PROJECT_ROOT / "data" / "reversal_data.csv"
DATA_DIR = PROJECT_ROOT / "data" / "iv_study"
CONTROLS_DIR = DATA_DIR / "controls"
REPORTS_DIR = PROJECT_ROOT / "reports" / "iv_study"
MANIFEST_CSV = DATA_DIR / "manifest.csv"

# Expiration selection: nearest listed expiry >= MIN_DTE, closest to TARGET_DTE.
# 0-2 DTE gamma/pin and expiry-day IV artifacts contaminate the signal.
MIN_DTE = 5
TARGET_DTE = 7
MAX_DTE = 45
EXP_CANDIDATES = 3          # retry with next-nearest expirations if greeks come back empty

# Strike ladder: strikes spanning the day's likely range so ATM IV can be
# reconstructed at constant moneyness as the underlying traverses 20-40%.
LADDER_N = 7
LADDER_LO = 0.75            # x snapshot spot
LADDER_HI = 1.25

INTERVAL = "1m"
SNAPSHOT_TIME = "09:35:00"  # chain snapshot for strike discovery / parity spot

# Bar-quality filters (per Theta probe: iv_error ~0 good, 100 failed solve;
# the 09:30 opening bar is junk even on liquid names).
IV_ERR_MAX = 99.0
MIN_VALID_FRAC = 0.70       # required fraction of session minutes with a valid ATM IV

# Event-study window (minutes relative to time_of_high_price) and z-score baseline.
WIN_PRE = 60
WIN_POST = 30
BASELINE = (-60, -30)

# Track B within-ticker pseudo-controls: same ticker, N trading days before the
# trade. 8 days gives room to measure the multi-day IV ramp into the reversal
# day (the core lead-up hypothesis), not just the final open gap.
PSEUDO_CONTROL_DAYS = 8

# Microcaps with no usable options data (verified: CODX returned nothing at all).
MICROCAP_DROPLIST = {
    "CODX", "AHPI", "APT", "USWS", "INDO", "HUSA", "USEG", "CEI", "IMPP", "MULN",
}

# Phase 1 pilot: (ticker, YYYY-MM-DD). Five post-10am RTH tops + one premarket
# topper (MRNA) to exercise the Track B-only path.
PILOT_TRADES = [
    ("GME", "2021-03-10"),
    ("NVDA", "2024-02-12"),
    ("SMCI", "2024-02-12"),
    ("BBBY", "2021-01-25"),
    ("MRNA", "2020-02-27"),
    ("NVDA", "2024-03-08"),
]
