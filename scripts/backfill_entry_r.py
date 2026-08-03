"""Entry-anchored magnitude scale for bounce signals.

Why
---
`tradeable_3d` measures MFE from the D0 OPEN. That misses the setup this system
actually trades: a gap-down that flushes BELOW the open and reverses scores a
negative MFE even when buying the flush was profitable. Low-anchoring is not the
fix either — over a 4-day window every name's low-to-high range clears 0.5 ATR,
so it grades 100% of signals tradeable and carries no information.

The honest anchor is the price the rules would actually have filled at. This
replays the live morning-watcher entry rule bar by bar and scores each signal by
how big a bounce it produced, in R off the real stop. Continuous, not binary:
bigger bounce -> bigger number.

Faithful to orderPipe/morning_watcher/rules/entry_bar_rules.py
--------------------------------------------------------------
  * entry = first 2-min bar CLOSING above the prior completed bar's high
  * LOD-RECENCY GATE: the low of day must have been set within LOD_RECENCY_WIN
    windows of the signal bar. This is NOT in bounce_entry_study.detectors.pbb_up
    — it was added live on 2026-07-28 because without it every up-bar in an
    ordinary uptrend "closes above the prior bar's high" and the rule fires
    mid-day, long after the bounce. Replaying without it would score a rule the
    trader does not use.
  * stop = LOD at signal time; one re-entry allowed after a stop-out; max 2/day
  * 6-minute cooldown between confirmed signals; nothing fresh after 15:30
  * risk floored at RISK_FLOOR_ATR ATRs (the study's robustness view — tiny
    stops otherwise inflate R)

Scored on D0 only, matching the rule's own same-day protocol (9:30 arm, LOD
recency, done by 15:30). NOTE: the scanner is known to fire early, so a D0-only
window under-credits signals that set up on D+1/D+2 — revisit with a re-arming
multi-day variant once this baseline is established.

Adds columns, never replaces: `entry_r` (best MFE in R across the day's
attempts), `entry_signals` (how many fired), `entry_price`, `entry_stop`.
`tradeable_3d` is untouched — 15 analysis cycles depend on its definition.

Usage:
    python -m scripts.backfill_entry_r --limit 25 --dry-run
    python -m scripts.backfill_entry_r
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bounce_entry_study.detectors import build_2min  # noqa: E402
from bounce_entry_study.exits import ExitSpec, simulate_day_exits  # noqa: E402
from bounce_entry_study.fetch_bars import fetch_day  # noqa: E402

OUTCOMES_FILE = PROJECT_ROOT / "data" / "signal_outcomes.csv"

# --- constants mirrored from entry_bar_rules.py -----------------------------
LOD_RECENCY_WIN = 3        # windows; a break only counts if it follows a fresh low
MAX_ENTRY_SIGNALS = 2      # first entry + one re-entry after a stop-out
COOLDOWN_BARS = 3          # 6 minutes at 2 min/bar
LAST_ENTRY = pd.Timestamp("1900-01-01 15:30").time()
RISK_FLOOR_ATR = 0.5       # study's robustness floor

# entry_r        = MFE in R off the ACTUAL entry-to-stop risk
# entry_r_floored= same, with risk floored at 0.5 ATR (the study's robustness
#                  view: tiny stops otherwise inflate R). Both are logged because
#                  the floor materially moves the scale — the entry study reported
#                  +3.2R/day raw vs +1.9R/day floored — and which one is the right
#                  lens is a judgement for the analysis, not for this script.
# exit_r = REALISED R under the trader's live exit rule (trail_exit_rules.py):
#   LOD disaster stop first, arm at entry + 1.5 ATR, then trail the prior 2-min
#   bar's low. This is the number that answers "what would I actually have
#   banked", as opposed to entry_r's best-case excursion.
ARM_ATR = 1.5
NEW_COLUMNS = ["entry_r", "entry_r_floored", "exit_r", "entry_signals",
               "entry_price", "entry_stop"]


def replay_entry_day(two: pd.DataFrame, atr_abs: float | None) -> dict:
    """Replay the live entry rule over one day's 2-min frame.

    Returns best MFE in R across the day's attempts (the "how big was the
    bounce" scale), plus the first attempt's fill and stop for traceability.
    """
    out = {"entry_r": "", "entry_r_floored": "", "exit_r": "", "entry_signals": 0,
           "entry_price": "", "entry_stop": ""}
    if two is None or two.empty or len(two) < 2:
        return out
    fire_times = []          # every LOD-gated break, for the exit simulator
    ends = two.index + pd.Timedelta(minutes=2)

    highs, lows, closes = two["high"].values, two["low"].values, two["close"].values
    n = len(two)
    starts = two.index

    day_low, day_low_win = lows[0], 0
    live_stop = None          # not None => an entry is live; blocks re-arming
    last_confirm = -99
    attempts = []

    for i in range(1, n):
        # -- LOD tracking (a break only counts when it follows a fresh low) --
        if lows[i] < day_low:
            day_low, day_low_win = lows[i], i

        # -- stop-out watch: strict new low below the live stop re-arms --
        if live_stop is not None and lows[i] < live_stop:
            live_stop = None

        armed = live_stop is None and len(attempts) < MAX_ENTRY_SIGNALS
        lod_recent = (i - day_low_win) <= LOD_RECENCY_WIN
        cooled = (i - last_confirm) >= COOLDOWN_BARS
        in_time = starts[i].time() < LAST_ENTRY

        if not (lod_recent and in_time):
            continue
        if closes[i] <= highs[i - 1]:      # must CLOSE above the prior bar's high
            continue
        # Every qualifying break feeds the exit simulator, which models the
        # position lifecycle itself (re-arm after a losing exit). The arming /
        # cooldown gates below are the ALERT protocol and only shape entry_r.
        fire_times.append(ends[i])

        if not (armed and cooled):
            continue

        entry = float(closes[i])
        stop = float(day_low)
        risk = entry - stop
        if risk <= 0:
            continue
        risk_floored = max(risk, RISK_FLOOR_ATR * atr_abs) if (atr_abs and atr_abs > 0) else risk

        # -- excursion from this fill until the stop is taken out --
        # If a bar both makes a new high and breaks the stop, assume the stop
        # came first: exclude that bar's high. Conservative on purpose.
        mfe = entry
        for j in range(i + 1, n):
            if lows[j] < stop:
                break
            mfe = max(mfe, float(highs[j]))

        attempts.append({
            "r": (mfe - entry) / risk,
            "r_floored": (mfe - entry) / risk_floored,
            "entry": entry, "stop": stop,
        })
        last_confirm = i
        live_stop = stop

    if attempts:
        out["entry_signals"] = len(attempts)
        out["entry_r"] = round(max(a["r"] for a in attempts), 3)
        out["entry_r_floored"] = round(max(a["r_floored"] for a in attempts), 3)
        out["entry_price"] = round(attempts[0]["entry"], 4)
        out["entry_stop"] = round(attempts[0]["stop"], 4)

    # Realised R under the live exit rule. Capped at MAX_ENTRY_SIGNALS attempts
    # to match the alert protocol's one-re-entry limit.
    if fire_times:
        try:
            res = simulate_day_exits(two, fire_times,
                                     ExitSpec(kind="trail", arm_atr=ARM_ATR), atr_abs)
            taken = res.attempts[:MAX_ENTRY_SIGNALS]
            if taken:
                out["exit_r"] = round(sum(a.r for a in taken), 3)
        except Exception:  # noqa: BLE001 - exit sim must not sink the row's MFE
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only process the first N bounce rows")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--refresh", action="store_true",
                    help="re-score rows that already have entry_r (bars are cached, so cheap)")
    args = ap.parse_args()

    with open(OUTCOMES_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
        header = list(rows[0].keys())

    todo = [r for r in rows if r.get("bucket") == "bounce"
            and str(r.get("complete")).lower() == "true"
            and (args.refresh or not str(r.get("entry_r") or "").strip())]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} bounce rows to score (of {len(rows)} total)\n")

    scored = fired = failed = 0
    for r in todo:
        ticker, date = r["ticker"], r["target_date"]
        try:
            bars = fetch_day(ticker, date)
            if bars is None or bars.empty:
                failed += 1
                continue
            two = build_2min(bars)
            atr_pct = float(r["atr_pct"]) if r.get("atr_pct") else None
            entry_open = float(r["entry_open"]) if r.get("entry_open") else None
            atr_abs = atr_pct * entry_open if (atr_pct and entry_open) else None
            res = replay_entry_day(two, atr_abs)
        except Exception as e:  # noqa: BLE001 - one bad day must not stop the sweep
            print(f"  {ticker} {date}: {e}")
            failed += 1
            continue

        r.update(res)
        scored += 1
        if res["entry_signals"]:
            fired += 1
        if scored % 100 == 0:
            print(f"  ...{scored}/{len(todo)}")

    print(f"\nscored {scored}, no bars {failed}")
    print(f"produced a valid entry signal: {fired}/{scored} "
          f"({fired / max(1, scored) * 100:.1f}%) — the rest had no qualifying "
          f"break after a fresh low, i.e. nothing to trade")

    if args.dry_run:
        vals = sorted(float(r["entry_r"]) for r in todo
                      if str(r.get("entry_r") or "").strip())
        if vals:
            print(f"\nentry_r on fired signals: median {vals[len(vals)//2]:.2f}R  "
                  f"p25 {vals[len(vals)//4]:.2f}R  p75 {vals[int(len(vals)*.75)]:.2f}R  "
                  f"max {vals[-1]:.2f}R")
        print("\n--dry-run: nothing written")
        return 0

    out_header = header + [c for c in NEW_COLUMNS if c not in header]
    tmp = OUTCOMES_FILE.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_header)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in out_header})
    tmp.replace(OUTCOMES_FILE)
    print(f"\nwrote {OUTCOMES_FILE.name} (+{len(NEW_COLUMNS)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
