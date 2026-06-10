"""Episode-level retest of proposed signal rules against logged outcomes.

Replays every logged reversal episode (data/signal_outcomes.csv) through
candidate rule sets and compares 1-ATR-in-3-days hit rates, measured BY
EPISODE (first qualifying entry per episode — no double-counting reprints).

Rules tested:
  BASELINE      enter on the episode's first flag (what the scanner did)
  VETO          skip flags with prior_day_rvol < 1.25; enter on the first
                surviving flag of the episode (deployed 2026-06-10)
  RESTRUCTURE   VETO + score restructure: drop the gap_pct point, add an
                RVOL-tier point (prior_day_rvol >= 2.0); enter on first
                flag scoring GO (>=4/5) under the new score
  REPRINT       enter only on a reprint whose prior print had already hit
                its 1-ATR target BEFORE the reprint's day (observable at
                entry time)
  PLAYBOOK      VETO entry, upgraded by REPRINT: small at first surviving
                flag, full size at paid-reprint trigger — measured here as
                "episode pays from whichever entry came first"

Also reports: Large-cap before/after veto, and the 4/5-vs-5/5 inversion
decomposition (does removing sub-1.25-RVOL + gap-credit explain it?).

Caveat printed with results: rules were derived from this same sample
(Analysis #2), so this is internal-consistency validation, not
out-of-sample proof. A first-half/second-half stability split is included
to show whether the veto effect holds across time. The true forward test
is the live VETO cohort tracking in the scorecard.

Usage:
    python scripts/retest_signal_rules.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTCOMES_FILE = PROJECT_ROOT / "data" / "signal_outcomes.csv"

RVOL_VETO = 1.25
RVOL_TIER = 2.0
PM_RVOL_THRESHOLD = 0.05  # mirrors generate_report._PREMARKET_RVOL_THRESHOLD


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_episodes() -> list[list[dict]]:
    """Complete reversal rows grouped into episodes, ordered by signal number."""
    with open(OUTCOMES_FILE, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["bucket"] == "reversal"
                and str(r.get("complete")).lower() == "true"
                and r.get("days_available") not in ("", "0")]
    eps = defaultdict(list)
    for r in rows:
        eps[r["episode_id"]].append(r)
    out = []
    for ep in eps.values():
        ep.sort(key=lambda r: int(float(r["episode_signal_num"] or 1)))
        out.append(ep)
    out.sort(key=lambda ep: ep[0]["target_date"])
    return out


def hit(r) -> bool:
    return str(r.get("tradeable_3d")).lower() == "true"


def vetoed(r) -> bool:
    rv = fnum(r.get("prior_day_rvol"))
    return rv is not None and rv < RVOL_VETO


def restructured_go(r) -> bool:
    """Recompute the 5-point score with gap_pct replaced by an RVOL tier point.

    Criteria thresholds come from ReversalScorer per cap. GO = score >= 4.
    Rows missing a feature fail that criterion (same as live scoring).
    """
    thr = _thresholds(r.get("cap", "Medium"))
    if thr is None:
        return False
    score = 0
    for key, t in (("pct_from_9ema", thr.pct_from_9ema),
                   ("prior_day_range_atr", thr.prior_day_range_atr),
                   ("pct_change_3", thr.pct_change_3)):
        v = fnum(r.get(key))
        if v is not None and v >= t:
            score += 1
    # vol signal (unchanged): prior-day RVOL >= cap threshold OR premarket RVOL
    pr, pm = fnum(r.get("prior_day_rvol")), fnum(r.get("premarket_rvol"))
    if (pr is not None and pr >= thr.rvol_score) or (pm is not None and pm >= PM_RVOL_THRESHOLD):
        score += 1
    # NEW: RVOL tier point replaces the gap_pct point
    if pr is not None and pr >= RVOL_TIER:
        score += 1
    return score >= 4


_THR_CACHE: dict = {}


def _thresholds(cap: str):
    if cap not in _THR_CACHE:
        try:
            from analyzers.reversal_scorer import ReversalScorer
            if "_scorer" not in _THR_CACHE:
                _THR_CACHE["_scorer"] = ReversalScorer()
            _THR_CACHE[cap] = _THR_CACHE["_scorer"]._get_thresholds(cap or "Medium")
        except Exception as e:
            print(f"(threshold load failed for cap={cap}: {e})")
            _THR_CACHE[cap] = None
    return _THR_CACHE[cap]


def prior_print_paid_before(ep: list[dict], idx: int) -> bool:
    """True if any earlier print in the episode hit its 1-ATR target on a day
    strictly before this print's target date (i.e. observable at entry)."""
    entry_date = ep[idx]["target_date"]
    for prev in ep[:idx]:
        d = prev.get("days_to_1atr")
        if d == "" or d is None:
            continue
        hit_day_offset = int(float(d))
        # conservative: require the prior print's window day to be a date
        # strictly before the reprint's target date. Day offsets map to
        # trading days from the prior print's target; its hit date is at
        # most `offset` trading days after prev target, which is < entry
        # date iff prev_target advanced by offset is still earlier.
        # We approximate with date strings: prev hit observable if
        # prev.target_date < entry_date and offset <= (#days between) - 1.
        # Since prints inside an episode are <=3 trading days apart and
        # offsets are 0..3, use the simple sufficient check below.
        if prev["target_date"] < entry_date and hit_day_offset <= _tradedays_between(prev["target_date"], entry_date) - 1:
            return True
    return False


_TDAYS: list[str] | None = None


def _tradedays_between(a: str, b: str) -> int:
    """Trading days strictly between dates is (idx_b - idx_a)."""
    global _TDAYS
    if _TDAYS is None:
        import pandas_market_calendars as mcal
        sched = mcal.get_calendar("NYSE").schedule(start_date="2026-03-01", end_date="2026-12-31")
        _TDAYS = [d.strftime("%Y-%m-%d") for d in sched.index]

    def idx(d):
        for i, x in enumerate(_TDAYS):
            if x >= d:
                return i
        return len(_TDAYS)

    return idx(b) - idx(a)


def rate(pairs: list[bool]) -> str:
    n = len(pairs)
    if n == 0:
        return "n=0"
    k = sum(pairs)
    se = (k / n * (1 - k / n) / n) ** 0.5 * 100 if n else 0
    return f"{k}/{n} = {k / n * 100:.1f}%  (±{se:.0f}pp)"


def run_rule(episodes, entry_fn, label) -> dict:
    """entry_fn(ep) -> index of entry print or None (episode skipped)."""
    results, skipped_hits = [], []
    for ep in episodes:
        i = entry_fn(ep)
        if i is None:
            skipped_hits.append(any(hit(r) for r in ep))
        else:
            results.append(hit(ep[i]))
    return {"label": label, "results": results, "skipped": skipped_hits}


def main():
    episodes = load_episodes()
    n_sig = sum(len(ep) for ep in episodes)
    print(f"\n{'=' * 74}\nEPISODE-LEVEL RULE RETEST — {len(episodes)} reversal episodes, {n_sig} signals")
    print(f"{'=' * 74}")
    print("Hit = signal's 1-ATR favorable target inside its D0..D+3 window.\n")

    def baseline_entry(ep):
        return 0

    def veto_entry(ep):
        for i, r in enumerate(ep):
            if not vetoed(r):
                return i
        return None

    def restructure_entry(ep):
        for i, r in enumerate(ep):
            if not vetoed(r) and restructured_go(r):
                return i
        return None

    def reprint_entry(ep):
        for i in range(1, len(ep)):
            if prior_print_paid_before(ep, i):
                return i
        return None

    def playbook_entry(ep):
        v = veto_entry(ep)
        rp = reprint_entry(ep)
        if v is None:
            return rp
        if rp is None:
            return v
        return min(v, rp)

    rules = [
        run_rule(episodes, baseline_entry, "BASELINE   first flag, no filter"),
        run_rule(episodes, veto_entry, "VETO       first flag with prior RVOL >= 1.25"),
        run_rule(episodes, restructure_entry, "RESTRUCT   veto + gap point -> RVOL>=2.0 tier, GO only"),
        run_rule(episodes, reprint_entry, "REPRINT    reprint after observable 1-ATR prior print"),
        run_rule(episodes, playbook_entry, "PLAYBOOK   earlier of VETO entry / REPRINT trigger"),
    ]

    print(f"{'rule':58s} {'hit rate (by episode)':26s} skipped eps (would-have-hit)")
    for r in rules:
        sk = f"{len(r['skipped'])} ({sum(r['skipped'])} hit)" if r["skipped"] else "0"
        print(f"  {r['label']:56s} {rate(r['results']):26s} {sk}")

    # ── Stability split (rule was derived on this sample — check both halves) ──
    half = len(episodes) // 2
    for name, eps in (("first half", episodes[:half]), ("second half", episodes[half:])):
        b = run_rule(eps, baseline_entry, "")
        v = run_rule(eps, veto_entry, "")
        print(f"\n  stability [{name}]  baseline {rate(b['results'])}   veto {rate(v['results'])}")

    # ── Large-cap before/after veto ──
    large = [ep for ep in episodes if ep[0].get("cap") == "Large"]
    if large:
        b = run_rule(large, baseline_entry, "")
        v = run_rule(large, veto_entry, "")
        print(f"\nLarge-cap episodes (n={len(large)}): baseline {rate(b['results'])} -> veto {rate(v['results'])}")

    # ── 4/5 vs 5/5 inversion decomposition (first flags) ──
    print("\n4/5 vs 5/5 inversion (first flags):")
    firsts = [ep[0] for ep in episodes]
    for sc in ("4/5", "5/5"):
        grp = [r for r in firsts if r.get("score") == sc]
        grp_v = [r for r in grp if not vetoed(r)]
        gap_credit = [r for r in grp_v if (fnum(r.get("gap_pct")) or 0) >= 0.0]
        print(f"  score {sc}: all {rate([hit(r) for r in grp])} | "
              f"post-veto {rate([hit(r) for r in grp_v])}")

    print("\nCAVEAT: rules were derived from this same sample (Analysis #2) — this")
    print("retest shows internal consistency + temporal stability, not out-of-sample")
    print("proof. Forward verification = live VETO cohort tracking in the scorecard.")


if __name__ == "__main__":
    main()
