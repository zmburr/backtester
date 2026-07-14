"""Configuration for the de-SPAC event study (2020-present)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "despac_study"
CACHE_DIR = DATA_DIR / "cache"
EDGAR_CACHE_DIR = DATA_DIR / "edgar_cache"
REPORTS_DIR = PROJECT_ROOT / "reports" / "despac_study"

for _d in (DATA_DIR, CACHE_DIR, EDGAR_CACHE_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

UNIVERSE_CSV = DATA_DIR / "despac_universe.csv"
TIMELINE_CSV = DATA_DIR / "despac_timeline.csv"
ENRICHED_CSV = DATA_DIR / "despac_enriched.csv"
CLASSIFIED_CSV = DATA_DIR / "despac_classified.csv"
MASTER_CSV = DATA_DIR / "despac_data.csv"
REPORT_HTML = REPORTS_DIR / "despac_report.html"

# SEC asks for <=10 req/s and a descriptive User-Agent.
EDGAR_UA = "zmburr despac research zmburr@gmail.com"
EDGAR_MIN_INTERVAL = 0.15  # ~6.6 req/s, polite margin

STUDY_START = "2020-01-01"

# Exact phrases that appear in completion ("super") 8-Ks. Swept per-quarter,
# unioned, then post-filtered to hits whose 8-K item list includes 2.01
# (Completion of Acquisition). Overlap between phrases is fine - deduped by
# accession number.
FTS_PHRASES = [
    '"previously announced business combination"',
    '"consummation of the business combination"',
    '"consummated the business combination"',
    '"completed the business combination"',
    '"closing of the business combination"',
    # some super 8-Ks (e.g. CCIV/Lucid) say "merger", never "business combination"
    '"previously announced merger"',
    '"consummation of the previously announced transactions"',
    # highest-recall SPAC marker: every completion 8-K describes trust-account
    # redemptions; conventional M&A never does
    '"trust account"',
]

# FTS pagination: 10 hits/page, from+size capped at 10000. If a window's
# total exceeds this, the sweep bisects the date range.
FTS_MAX_HITS = 9500

# Event-study windows (trading days)
PRICE_PAD_DAYS = 15
POST_CLOSE_DAYS = [1, 2, 3, 5, 10]

# Sector buckets for classification
SECTOR_BUCKETS = [
    "ev_mobility", "space_defense", "fintech", "biotech_health",
    "software_ai", "crypto", "energy", "industrial", "consumer",
    "media_gaming", "real_estate", "other",
]
HIGHTECH_BUCKETS = {"ev_mobility", "space_defense", "fintech", "software_ai", "crypto"}

# Non-SPAC reorgs that leak through the text-evidence filter with unrecorded
# exchange ratios (iStar->Safehold 0.16x, NCR spinoff, AEL preferred stub)
KNOWN_NON_SPAC_FLIPS = {"SAFE", "VYX", "ANGpA"}

# Known deals for verification (old ticker, new ticker, approx close date)
SPOT_CHECKS = [
    ("VTIQ", "NKLA", "2020-06-03"),
    ("IPOE", "SOFI", "2021-05-28"),
    ("CCIV", "LCID", "2021-07-23"),
    ("GGPI", "PSNY", "2022-06-23"),
    ("DWAC", "DJT", "2024-03-25"),
]
