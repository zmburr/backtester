"""
Validate the archetype gate (Option 1) against reversal_data.csv before deploy.

Acceptance criteria:
  - <= 5-10% of A-grade reversals flagged (gate is permissive, only catches
    genuine off-archetype setups)
  - Mean open->low of flagged A-grades is NOT better than non-flagged A-grades
    (i.e. the gate isn't eating real winners)
  - Recent priority-report JSONs (if present) show IONQ-type signals flagged
    and CAR-type passed

Run:
    python scripts/validate_archetype_gate.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyzers.reversal_scorer import (
    check_archetype,
    ARCHETYPE_PCT_FROM_200MAV,
    ARCHETYPE_PCT_CHANGE_30,
)

REVERSAL_CSV = ROOT / "data" / "reversal_data.csv"
SIGNAL_DIR = ROOT / "data" / "signals"  # priority_report JSON dumps (if any)


def validate_historical():
    df = pd.read_csv(REVERSAL_CSV)
    print(f"Loaded {len(df)} rows from reversal_data.csv")
    print(f"Archetype thresholds: pct_from_200mav > {ARCHETYPE_PCT_FROM_200MAV}, "
          f"pct_change_30 > {ARCHETYPE_PCT_CHANGE_30}, or breaks_52wk\n")

    results = []
    for _, row in df.iterrows():
        passed, detail = check_archetype(row.to_dict())
        results.append({
            "date": row.get("date"),
            "ticker": row.get("ticker"),
            "grade": row.get("trade_grade"),
            "cap": row.get("cap"),
            "passed": passed,
            "by_52wk": detail["passed_by_52wk"],
            "by_200mav": detail["passed_by_200mav"],
            "by_30d": detail["passed_by_30d"],
            "pct_from_200mav": detail["pct_from_200mav"],
            "pct_change_30": detail["pct_change_30"],
            "breaks_fifty_two_wk": detail["breaks_fifty_two_wk"],
            "open_close_pct": row.get("reversal_open_close_pct"),
            "open_low_pct": row.get("reversal_open_low_pct"),
        })
    r = pd.DataFrame(results)

    print("=" * 70)
    print("Flag rate by grade")
    print("=" * 70)
    for g in ["A", "B", "C", "D"]:
        sub = r[r["grade"] == g]
        if len(sub) == 0:
            continue
        flagged = (~sub["passed"]).sum()
        pct = flagged / len(sub) * 100
        print(f"  grade={g}  n={len(sub):3d}  flagged={flagged:3d} ({pct:.1f}%)")
    print()

    print("=" * 70)
    print("Which sub-rule saved each A-grade (i.e. why it passed)")
    print("=" * 70)
    a = r[r["grade"] == "A"]
    print(f"  passed via breaks_52wk:  {a['by_52wk'].sum()}/{len(a)}")
    print(f"  passed via pct_200mav:   {a['by_200mav'].sum()}/{len(a)}")
    print(f"  passed via pct_change_30:{a['by_30d'].sum()}/{len(a)}")
    print()

    print("=" * 70)
    print("Flagged A-grades (these would be downgraded GO -> CAUTION)")
    print("=" * 70)
    a_flagged = a[~a["passed"]]
    if len(a_flagged) == 0:
        print("  (none)")
    else:
        cols = ["date", "ticker", "cap", "pct_from_200mav", "pct_change_30",
                "breaks_fifty_two_wk", "open_close_pct", "open_low_pct"]
        print(a_flagged[cols].to_string(index=False))
    print()

    print("=" * 70)
    print("Outcome comparison — A-grades: flagged vs passed")
    print("(NEGATIVE open_close/open_low = profitable short)")
    print("=" * 70)
    if len(a_flagged) > 0:
        print(f"  Flagged A-grade  n={len(a_flagged):3d}  "
              f"mean open_close={a_flagged['open_close_pct'].mean():+.3f}  "
              f"mean open_low={a_flagged['open_low_pct'].mean():+.3f}")
    a_passed = a[a["passed"]]
    print(f"  Passed  A-grade  n={len(a_passed):3d}  "
          f"mean open_close={a_passed['open_close_pct'].mean():+.3f}  "
          f"mean open_low={a_passed['open_low_pct'].mean():+.3f}")
    print()

    # Acceptance summary
    print("=" * 70)
    print("ACCEPTANCE CHECK")
    print("=" * 70)
    a_flag_rate = (len(a_flagged) / len(a) * 100) if len(a) else 0
    a_flag_mfe = a_flagged["open_low_pct"].mean() if len(a_flagged) else 0
    a_pass_mfe = a_passed["open_low_pct"].mean() if len(a_passed) else 0
    ok_flag_rate = a_flag_rate <= 10.0
    # Winners mean MORE negative MFE (they short further). So for the gate to be
    # NOT eating real winners, flagged A-grade MFE should be less negative
    # (smaller move) than passed A-grade MFE.
    ok_not_eating = a_flag_mfe >= a_pass_mfe if len(a_flagged) else True

    print(f"  A-grade flag rate: {a_flag_rate:.1f}% (target <= 10%): "
          f"{'PASS' if ok_flag_rate else 'FAIL'}")
    print(f"  Flagged A-grade MFE {a_flag_mfe:+.3f} vs passed {a_pass_mfe:+.3f}: "
          f"{'PASS' if ok_not_eating else 'FAIL'} (flagged should be less negative)")
    print()
    return r


def validate_recent_signals():
    print("=" * 70)
    print("Replay recent priority-report signals (if present)")
    print("=" * 70)
    if not SIGNAL_DIR.exists():
        print(f"  {SIGNAL_DIR} does not exist — skipping")
        return
    files = sorted(SIGNAL_DIR.glob("*.json"))[-10:]
    if not files:
        print("  no signal JSONs found — skipping")
        return

    for f in files:
        try:
            payload = json.loads(f.read_text())
        except Exception as e:
            print(f"  {f.name}: read error {e}")
            continue
        for sig in payload.get("signals", []):
            if sig.get("bucket") != "reversal":
                continue
            m = sig.get("metrics", {}) or {}
            passed, detail = check_archetype(m)
            status = "PASS" if passed else "FLAG"
            rec = sig.get("recommendation", "?")
            print(f"  {payload.get('date','?')} {sig.get('ticker','?'):6s} "
                  f"{rec:7s}  archetype={status}  "
                  f"p200={detail['pct_from_200mav']}  "
                  f"p30={detail['pct_change_30']}  "
                  f"52w={detail['breaks_fifty_two_wk']}")


if __name__ == "__main__":
    validate_historical()
    validate_recent_signals()
