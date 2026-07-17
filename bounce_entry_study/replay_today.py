"""Replay the validated entry+exit system on today's watched tickers.

    python -m bounce_entry_study.replay_today [YYYY-MM-DD]
"""
from __future__ import annotations

import sys

from .confluence import confluence_fires
from .exits import ExitSpec, simulate_day_exits
from .run_study import prepare_days

ENTRY = {"dets": ("pbb:h0",), "k": 1, "window": 16, "gate": "09:30"}
EXITS = [ExitSpec("trail", arm_atr=1.5), ExitSpec("close"), ExitSpec("vwap")]


def run(date_iso: str | None) -> None:
    hist, today = prepare_days()
    days = today if date_iso is None else [d for d in hist + today if d.date_iso == date_iso]
    if not days:
        print(f"no cached days for {date_iso}")
        return

    for d in days:
        atr = d.two.attrs.get("atr")
        fires = confluence_fires(d.two, {v: d.det_fires[v] for v in ENTRY["dets"]},
                                 k=ENTRY["k"], window_min=ENTRY["window"],
                                 gate=ENTRY["gate"])
        op = float(d.two["open"].iloc[0])
        lo_t = (d.two.index[int(d.two['low'].values.argmin())]).strftime("%H:%M")
        print(f"\n{'='*72}\n{d.ticker} {d.date_iso}  ({d.setup}, {d.cap})  "
              f"open {op:.2f}  ATR {atr:.2f}  day low {d.two['low'].min():.2f} @ {lo_t}  "
              f"close {d.two['close'].iloc[-1]:.2f}")
        for spec in EXITS:
            r = simulate_day_exits(d.two, fires, spec, atr)
            print(f"  [{spec.name}]  day total {r.total_r:+.2f}R")
            for a in r.attempts:
                arm = f"  (arm level {a.entry + 1.5*atr:.2f})" if spec.kind == "trail" and atr else ""
                how = "LOD STOP" if a.stopped else (
                    "held to close" if a.exit_time == (d.two.index[-1] + (d.two.index[1]-d.two.index[0]))
                    else "exit")
                print(f"      {a.entry_time.strftime('%H:%M')} in @ {a.entry:.2f} "
                      f"stop {a.stop:.2f}{arm} -> "
                      f"{a.exit_time.strftime('%H:%M')} out @ {a.exit:.2f} "
                      f"[{how}]  {a.r:+.2f}R")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
