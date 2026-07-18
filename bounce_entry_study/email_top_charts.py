"""Email candlestick charts of the top-N bounce days, system entries/exits marked.

    python -m bounce_entry_study.email_top_charts [--top 10] [--no-send]

Chart annotations (the validated system: first 2-min break entry, LOD stop,
re-entry after stop-outs, trail armed at +1.5 ATR):
  green ^  entry (at the fire bar close)
  red v    LOD stop-out
  blue o   final exit (trail hit or session close)
  red --   the LOD stop level of each attempt
  amber -- the +1.5 ATR arm level of the surviving attempt
  grey     session VWAP
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from support.config import send_email

from .confluence import confluence_fires
from .detectors import BAR_MIN
from .exits import ExitSpec, simulate_day_exits
from .run_study import REPORTS, prepare_days

log = logging.getLogger(__name__)
CHART_DIR = REPORTS / "charts"

ENTRY = {"dets": ("pbb:h0",), "k": 1, "window": 16, "gate": "09:30"}
EXIT = ExitSpec("trail", arm_atr=1.5)


def _chart(d, res, path: Path) -> None:
    import mplfinance as mpf

    two = d.two
    atr = two.attrs.get("atr")
    df = two.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    def marker_series(points):
        s = pd.Series(np.nan, index=df.index)
        for t, px in points:
            start = t - pd.Timedelta(minutes=BAR_MIN)
            if start in s.index:
                s.loc[start] = px
        return s

    entries = marker_series([(a.entry_time, a.entry) for a in res.attempts])
    stops = marker_series([(a.exit_time, a.exit) for a in res.attempts if a.stopped])
    finals = marker_series([(a.exit_time, a.exit) for a in res.attempts
                            if not a.stopped and a.exit_time is not None])

    aps = [mpf.make_addplot(two["vwap_at_end"].values, color="#8b949e", width=0.9)]
    for s, marker, color, size in ((entries, "^", "#3fb950", 130),
                                   (stops, "v", "#f85149", 130),
                                   (finals, "o", "#58a6ff", 110)):
        if s.notna().any():
            aps.append(mpf.make_addplot(s.values, type="scatter", marker=marker,
                                        color=color, markersize=size))

    hlines, hcolors, hstyles = [], [], []
    for a in res.attempts:
        hlines.append(a.stop)
        hcolors.append("#f85149")
        hstyles.append("--")
    survivor = next((a for a in res.attempts if not a.stopped), None)
    if survivor is not None and atr:
        arm = survivor.entry + 1.5 * atr
        # only draw the arm level when it's in play — an unreached level far
        # above the range would just compress the candles
        if arm <= float(two["high"].max()) * 1.02:
            hlines.append(arm)
            hcolors.append("#e3b341")
            hstyles.append(":")

    title = (f"{d.ticker} {d.date_iso}  ·  {d.setup} {d.cap}  ·  "
             f"{res.total_r:+.1f}R ({len(res.attempts)} attempt"
             f"{'s' if len(res.attempts) != 1 else ''})")
    mpf.plot(df, type="candle", style="nightclouds", addplot=aps,
             hlines=dict(hlines=hlines, colors=hcolors, linestyle=hstyles,
                         linewidths=[1.0] * len(hlines)),
             title=title, ylabel="", volume=False, figsize=(13, 4.6),
             savefig=dict(fname=str(path), dpi=110, bbox_inches="tight"))


def run(top_n: int, send: bool) -> None:
    hist, _ = prepare_days()
    results = []
    for d in hist:
        fires = confluence_fires(d.two, {v: d.det_fires[v] for v in ENTRY["dets"]},
                                 k=ENTRY["k"], window_min=ENTRY["window"],
                                 gate=ENTRY["gate"])
        results.append((d, simulate_day_exits(d.two, fires, EXIT, d.two.attrs.get("atr"))))
    results.sort(key=lambda x: x[1].total_r, reverse=True)
    top = results[:top_n]

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    inline, rows = {}, []
    for i, (d, res) in enumerate(top, 1):
        path = CHART_DIR / f"top{i:02d}_{d.ticker}_{d.date_iso}.png"
        _chart(d, res, path)
        cid = f"chart{i}"
        inline[cid] = str(path)
        att = "; ".join(
            f"{a.entry_time.strftime('%H:%M')} in @{a.entry:.2f} -> "
            f"{a.exit_time.strftime('%H:%M')} out @{a.exit:.2f} "
            f"({'stop' if a.stopped else 'exit'} {a.r:+.1f}R)"
            for a in res.attempts)
        rows.append(
            f"<h3 style='margin:24px 0 2px'>#{i} — {d.ticker} {d.date_iso} "
            f"· {d.setup} · <b>{res.total_r:+.1f}R</b></h3>"
            f"<div style='color:#666;font-size:13px;margin-bottom:6px'>{att}</div>"
            f"<img src='cid:{cid}' style='max-width:100%'>")
        log.info(f"charted #{i}: {d.ticker} {d.date_iso} {res.total_r:+.1f}R")

    body = (
        "<html><body style='font-family:Segoe UI,Arial,sans-serif'>"
        "<h2>Top {} bounce trades — new entry/exit system</h2>"
        "<p>Entry: first 2-min close over prior high (green ^), stop at LOD "
        "(red dashed), re-entry after stop-outs (red v). Exit: trail armed at "
        "+1.5 ATR (amber dotted), prior 2-min low trail / session close "
        "(blue o). Grey line = session VWAP — note how often the old "
        "sell-at-VWAP habit would have exited within minutes.</p>{}"
        "</body></html>"
    ).format(len(top), "".join(rows))

    if send:
        send_email("zmburr@gmail.com",
                   f"Top {len(top)} bounce trades — system entries/exits charted",
                   body, is_html=True, inline_images=inline)
    else:
        log.info(f"--no-send: {len(inline)} charts in {CHART_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--no-send", action="store_true")
    args = ap.parse_args()
    run(args.top, send=not args.no_send)
