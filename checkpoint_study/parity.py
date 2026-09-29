"""Parity: rule_sim's first breach vs the REAL orderPipe TradeManager.

Feeds cached 5s bars (+ 2-min bars derived from them) through the live
DataHandler / TradeManager in stream order — 5s first on close-time ties, the
last completed 2-min bar first as the on-connect snapshot — calling
process_snapshot() synchronously after every 5s bar. Records the first time
trade_watcher.stop_alerted_price is set. Audio muted, exit chart disabled,
volume watcher detached (it would fetch ADV over the network).

    orderPipe\\venv\\Scripts\\python -m checkpoint_study.parity --n 30
"""
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from checkpoint_study import config

os.environ.setdefault("EXIT_CHART", "0")


def _messages(fives: pd.DataFrame, twos: pd.DataFrame, anchor, end, symbol):
    def msg(typ, ts, r):
        return {"type": typ, "close-time": int(ts.value), "symbol": symbol,
                "open": float(r.open), "high": float(r.high), "low": float(r.low),
                "close": float(r.close), "volume": float(r.volume), "vwap": float(r.close)}
    out = []
    snap = twos[twos.index <= anchor]
    if len(snap):
        out.append(msg("bar-2min", snap.index[-1], snap.iloc[-1]))
    for typ, df in (("bar-5s", fives), ("bar-2min", twos)):
        w = df[(df.index > anchor) & (df.index <= end)]
        out += [msg(typ, ts, r) for ts, r in zip(w.index, w.itertuples())]
    out.sort(key=lambda m: (m["close-time"], m["type"] == "bar-2min"))
    return out


def live_first_breach(fives, twos, *, symbol, anchor, side, entry, ref, horizon_min=120,
                      strategy="2_min_quick"):
    config.orderpipe_on_path()
    import trader.trade_manager as tm
    import trader.trade_watcher as tw
    from calculators.stop_calculator import get_stop_offset
    from trader.trade import Trade
    from trillium.trlm_live_data import TrlmData

    quiet = lambda *a, **k: None
    tm.play_sounds_in_thread = tw.play_sounds_in_thread = tw.play_reward = quiet
    headline = anchor.strftime("%Y-%m-%d %H:%M:%S")
    trade = Trade(headline, symbol, "BUY" if side > 0 else "SELL", strategy)
    trade.position_size = 100 if side > 0 else -100
    trade.avg_price = entry
    trade.ref_price = ref
    trade.stop_multiplier = get_stop_offset(ref)
    manager = tm.TradeManager(trade, strategy)
    manager.data_handler.trade_watcher = None          # no ADV fetch / volume alerts
    end = min(anchor + pd.Timedelta(minutes=horizon_min),
              anchor.normalize() + pd.Timedelta(hours=16))
    msgs = _messages(fives, twos, anchor, end, symbol)
    first5 = next(m for m in msgs if m["type"] == "bar-5s")
    trade.last_price = first5["close"]
    manager.set_stop()
    feeder = TrlmData(trade, manager)
    for m in msgs:
        feeder._on_message(m)
        if m["type"] != "bar-5s":
            continue
        manager.process_snapshot(manager.data_handler.get_snapshot())
        manager.reset_for_next_iteration()
        if manager.trade_watcher.stop_alerted_price is not None:
            return pd.Timestamp(m["close-time"], tz="UTC").tz_convert(anchor.tz)
    return None


def run(n: int = 30, seed: int = 7, trail: str = "quick") -> pd.DataFrame:
    from checkpoint_study.fetch_bars import load_day
    from checkpoint_study.panel import _shared
    from checkpoint_study.population import build_episodes, live_ref
    from checkpoint_study.rule_sim import DayBars, SimParams, simulate

    feats, pf, ref_fn = _shared()
    extremes = lambda bars: pf.clean_extremes(bars)[:2]
    ep = build_episodes()
    ep = ep[ep["in_scope"]]
    have = ep[[load_day(s, d) is not None for s, d in zip(ep.symbol, ep.date)]]
    rows = []
    for e in have.sample(min(n, len(have)), random_state=seed).itertuples():
        fives = load_day(e.symbol, e.date)
        side, entry = int(e.side), float(e.entry)
        ref, _, _ = live_ref(fives, e.anchor, side, entry, ref_fn)
        sim = simulate(DayBars.build(fives, feats.minute_bars, extremes), anchor=e.anchor,
                       side=side, ref=ref, r_ps=1.0, p=SimParams(trail=trail))
        raw_twos = feats.minute_bars(fives, 2)      # "official" bars; DataHandler filters itself
        live = live_first_breach(fives, raw_twos, symbol=e.symbol, anchor=e.anchor,
                                 side=side, entry=entry, ref=ref,
                                 strategy="2_min_quick" if trail == "quick" else "2_min_close")
        s = sim["path"]["first_breach_ts"] if sim["path"] else None
        rows.append({"episode": e.episode_id, "sim": s, "live": live,
                     "match": (s is None and live is None) or
                              (s is not None and live is not None and abs((s - live).total_seconds()) <= 5)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--trail", default="quick", choices=["quick", "close"])
    a = ap.parse_args()
    df = run(a.n, trail=a.trail)
    print(df.to_string(index=False))
    print(f"\nmatch {int(df.match.sum())}/{len(df)} = {df.match.mean():.0%}")
