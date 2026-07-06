"""Enrich an existing priority-signal JSON with analogs / odds / bounce_cohort.

One-off for testing the morning watcher's bounce analog card against a past
day whose signal file predates the enrichment (the saved bounce metrics
already contain all six comp features, so the kNN can run retroactively).

Usage:
    python -m scripts.enrich_signal_file data/priority_signals/2026-07-06_morning.json \
        --tickers AAOI,AXTI -o data/priority_signals/2026-07-06_morning_enriched.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.priority_report import (
    BOUNCE_COMP_COLUMNS,
    _BOUNCE_DF,
    _analogs_payload,
    _bounce_cohort_payload,
    _json_default,
    _lookup_bounce_odds,
    find_historical_comps,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retro-enrich a priority signal JSON")
    parser.add_argument("signal_file", type=Path)
    parser.add_argument("--tickers", type=str, default=None,
                        help="comma-separated subset to keep (default: all)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="default: <input>_enriched.json")
    args = parser.parse_args()

    payload = json.loads(args.signal_file.read_text())
    keep = ({t.strip().upper() for t in args.tickers.split(",")}
            if args.tickers else None)

    signals = []
    for sig in payload.get("signals", []):
        if keep is not None and sig["ticker"].upper() not in keep:
            continue
        if sig.get("bucket") == "bounce":
            comps = find_historical_comps(sig.get("metrics", {}), _BOUNCE_DF,
                                          BOUNCE_COMP_COLUMNS, sig.get("cap", ""))
            analogs = _analogs_payload(comps)
            if analogs is not None:
                sig["analogs"] = analogs
            odds = _lookup_bounce_odds(sig.get("score", ""), sig.get("cap", ""))
            if odds is not None:
                sig["odds"] = odds
        signals.append(sig)

    payload["signals"] = signals
    if any(s.get("bucket") == "bounce" for s in signals):
        cohort = _bounce_cohort_payload()
        if cohort is not None:
            payload["bounce_cohort"] = cohort

    out = args.output or args.signal_file.with_name(args.signal_file.stem + "_enriched.json")
    out.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"Wrote {out} ({len(signals)} signals)")
    for s in signals:
        has = [k for k in ("analogs", "odds") if k in s]
        print(f"  {s['ticker']}: {', '.join(has) if has else 'no enrichment'}")


if __name__ == "__main__":
    main()
