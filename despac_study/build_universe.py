"""Stage 01: build the completed de-SPAC universe from EDGAR full-text search.

Sweeps quarter-by-quarter for 8-Ks containing completion-of-business-
combination phrases, keeps hits whose item list includes 2.01 (Completion of
Acquisition), and dedupes to one row per CIK (earliest completion 8-K).

Run:  python -m despac_study.build_universe [--start 2020-01-01] [--end YYYY-MM-DD]
"""

import argparse
import datetime as dt
import logging
import re

import pandas as pd

from despac_study.config import FTS_PHRASES, STUDY_START, UNIVERSE_CSV
from despac_study.edgar_client import fts_sweep

logger = logging.getLogger(__name__)

# "Trump Media & Technology Group Corp.  (DJT)  (CIK 0001849635)"
_NAME_RE = re.compile(r"^(.*?)\s*(?:\(([^)]*)\))?\s*\(CIK\s+(\d+)\)\s*$")


def _quarters(start: str, end: str):
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    cur = dt.date(s.year, 3 * ((s.month - 1) // 3) + 1, 1)
    while cur <= e:
        nxt = dt.date(cur.year + (cur.month + 3 > 12), (cur.month + 2) % 12 + 1, 1)
        q_end = min(nxt - dt.timedelta(days=1), e)
        yield max(cur, s).isoformat(), q_end.isoformat()
        cur = nxt


def parse_display_name(display: str):
    m = _NAME_RE.match(display or "")
    if not m:
        return display, "", ""
    name, tickers, cik = m.groups()
    return name.strip(), (tickers or "").strip(), cik


def build_universe(start: str, end: str) -> pd.DataFrame:
    rows = {}
    seen_adsh = set()
    for q_start, q_end in _quarters(start, end):
        q_hits = 0
        for phrase in FTS_PHRASES:
            for hit in fts_sweep(phrase, "8-K", q_start, q_end):
                src = hit.get("_source", {})
                adsh = src.get("adsh")
                if not adsh or adsh in seen_adsh:
                    continue
                seen_adsh.add(adsh)
                items = src.get("items") or []
                if "2.01" not in items:
                    continue
                q_hits += 1
                display = (src.get("display_names") or [""])[0]
                name, tickers, _ = parse_display_name(display)
                cik = (src.get("ciks") or [""])[0]
                if not cik:
                    continue
                cik = int(cik)
                row = {
                    "cik": cik,
                    "company_name": name,
                    "tickers": tickers,
                    "close_date": src.get("period_ending") or "",
                    "super8k_date": src.get("file_date") or "",
                    "super8k_adsh": adsh,
                    "items": ",".join(items),
                    "sic_at_filing": (src.get("sics") or [""])[0],
                    "form": src.get("form", ""),
                }
                prev = rows.get(cik)
                # keep the earliest completion 8-K per CIK (amendments and
                # follow-ups come later)
                if prev is None or row["super8k_date"] < prev["super8k_date"]:
                    rows[cik] = row
        logger.info("%s..%s: universe now %d CIKs (+%d item-2.01 hits)",
                    q_start, q_end, len(rows), q_hits)
    df = pd.DataFrame(list(rows.values())).sort_values("super8k_date")
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=STUDY_START)
    ap.add_argument("--end", default=dt.date.today().isoformat())
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = build_universe(args.start, args.end)
    df.to_csv(UNIVERSE_CSV, index=False)
    by_year = df["super8k_date"].str[:4].value_counts().sort_index()
    print(f"\nWrote {len(df)} completed de-SPACs -> {UNIVERSE_CSV}")
    print(by_year.to_string())


if __name__ == "__main__":
    main()
