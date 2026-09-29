"""Panel: one row per (episode, checkpoint) SURVIVOR — features + forward policy R.

    orderPipe\\venv\\Scripts\\python -m checkpoint_study.panel                # primary
    orderPipe\\venv\\Scripts\\python -m checkpoint_study.panel --variant floor_010

A variant changes one knob (R floor, anchor jitter, print filter, ref source,
latency, horizon, trail rule) and writes its own panel; analyze.py uses the
variants only as sign checks on the cells the primary run selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from checkpoint_study import config
from checkpoint_study.fetch_bars import load_day
from checkpoint_study.population import build_episodes, live_ref
from checkpoint_study.rule_sim import DayBars, SimParams, simulate

log = logging.getLogger(__name__)
ONE_R_DOLLARS = 3000.0     # ExitMonitor ONE_R today; no per-date vintage available here
WINSOR_R = 5.0


@dataclass(frozen=True)
class Variant:
    name: str = "primary"
    floor_pct: float = 0.0015
    atr_mult: float = 0.5
    anchor_jitter_s: int = 0
    filter_prints: bool = True
    ref_mode: str = "live"          # 'live' | 'logged'
    sim: SimParams = SimParams()


VARIANTS = {v.name: v for v in [
    Variant(),
    Variant("floor_010", floor_pct=0.0010),
    Variant("floor_025", floor_pct=0.0025),
    Variant("atr_0", atr_mult=0.0),
    Variant("atr_1", atr_mult=1.0),
    Variant("anchor_m30", anchor_jitter_s=-30),
    Variant("anchor_p30", anchor_jitter_s=30),
    Variant("no_filter", filter_prints=False),
    Variant("ref_logged", ref_mode="logged"),
    Variant("latency_30", sim=SimParams(latency_s=30)),
    Variant("horizon_60", sim=SimParams(horizon_min=60)),
    Variant("horizon_240", sim=SimParams(horizon_min=240)),
    Variant("trail_close", sim=SimParams(trail="close")),
]}


def _shared():
    feats = config.load_by_path("calculators/checkpoint_features.py", "op_checkpoint_features")
    pf = config.load_by_path("calculators/print_filter.py", "op_print_filter")
    config.orderpipe_on_path()
    from calculators.ref_price_calculator import get_reference_info_from_bars
    return feats, pf, get_reference_info_from_bars


def _sha(rel: str) -> str:
    return hashlib.sha256((config.ORDERPIPE_ROOT / rel).read_bytes()).hexdigest()[:12]


def build_panel(v: Variant = Variant(), limit_days: int = 0) -> tuple:
    feats, pf, ref_fn = _shared()
    extremes = (lambda bars: pf.clean_extremes(bars)[:2]) if v.filter_prints else None
    ep = build_episodes()
    ep = ep[ep["in_scope"]].copy()
    funnel = Counter(episodes=len(ep))
    rows, meta = [], []
    days = ep.groupby(["symbol", "date"], sort=True)
    for n_day, ((sym, date), grp) in enumerate(days):
        if limit_days and n_day >= limit_days:
            break
        fives = load_day(sym, date)
        if fives is None or fives.empty:
            funnel["no_bars"] += len(grp)
            continue
        day0 = pd.Timestamp(date, tz="US/Eastern")
        fives = fives[(fives.index > day0 + pd.Timedelta(hours=8, minutes=45))
                      & (fives.index <= day0 + pd.Timedelta(hours=16, minutes=5))]
        if fives.empty:
            funnel["no_bars"] += len(grp)
            continue
        day = DayBars.build(fives, feats.minute_bars, extremes)
        lo, hi = float(fives["low"].min()), float(fives["high"].max())
        for e in grp.itertuples(index=False):
            side, entry = int(e.side), float(e.entry)
            if not (lo * 0.98 <= entry <= hi * 1.02):
                funnel["price_mismatch"] += 1     # split-adjusted / wrong-symbol data
                continue
            anchor = e.anchor + pd.Timedelta(seconds=v.anchor_jitter_s)
            if v.ref_mode == "logged":
                ref = e.ref_logged
                if pd.isna(ref) or side * (entry - ref) <= 0:
                    funnel["no_logged_ref"] += 1
                    continue
                ref_src, guarded = "logged", False
            else:
                ref, ref_src, guarded = live_ref(fives, anchor, side, entry, ref_fn)
            pre = feats.pre_entry_ranges(fives, anchor, extremes)
            r_ps, r_src = feats.r_unit(entry, ref, pre, floor_pct=v.floor_pct, atr_mult=v.atr_mult)
            if not r_ps > 0:
                funnel["zero_r"] += 1
                continue
            res = simulate(day, anchor=anchor, side=side, ref=float(ref), r_ps=r_ps, p=v.sim,
                           checkpoints=config.CHECKPOINTS,
                           features_fn=feats.extract_features, entry=entry)
            if res["path"] is None:
                funnel["no_post_bars"] += 1
                continue
            funnel["simulated"] += 1
            funnel[f"first_breach_{res['path']['first_breach_phase']}"] += 1
            meta.append({"episode_id": e.episode_id, "ref": ref, "ref_src": ref_src,
                         "ref_guarded": guarded, "r_ps": r_ps, "r_src": r_src,
                         "r_pct": r_ps / entry, **{k: v2 for k, v2 in res["path"].items()}})
            for k, row in res["k"].items():
                if row is None:
                    continue
                funnel[f"survivor_{k}"] += 1
                rows.append({"episode_id": e.episode_id, "symbol": sym, "date": date,
                             "split": e.split, "side": side, "k": k, "entry": entry,
                             "ref": ref, "ref_guarded": guarded, "r_ps": r_ps, "r_src": r_src,
                             "max_size": e.max_size, "mkt_cap": e.mkt_cap,
                             "tod_min": anchor.hour * 60 + anchor.minute, **row})
    panel = pd.DataFrame(rows)
    if not panel.empty:
        for pol in ("hold", "trail_1m", "exit_now", "trim_half"):
            panel[f"fwd_{pol}"] = panel[f"fwd_{pol}"].clip(-WINSOR_R * 3, WINSOR_R * 3)
        for pol in ("trail_1m", "exit_now", "trim_half"):
            panel[f"edge_{pol}"] = (panel[f"fwd_{pol}"] - panel["fwd_hold"]).clip(-WINSOR_R, WINSOR_R)
        # Account-R: the same forward move on the position actually held, in 1R units.
        panel["acct_edge_exit_now"] = (panel["edge_exit_now"] * panel["r_ps"]
                                       * panel["max_size"] / ONE_R_DOLLARS)
    info = {"variant": v.name, "params": json.loads(json.dumps(asdict(v), default=str)),
            "feature_version": feats.FEATURE_VERSION, "checkpoint_features_sha": _sha("calculators/checkpoint_features.py"),
            "print_filter_sha": _sha("calculators/print_filter.py"), "funnel": dict(funnel),
            "built": pd.Timestamp.now(tz="US/Eastern").isoformat()}
    return panel, pd.DataFrame(meta), info


def save(panel, meta, info) -> None:
    out = config.DATA_DIR / "panels"
    out.mkdir(parents=True, exist_ok=True)
    panel.to_pickle(out / f"{info['variant']}.pkl")
    meta.to_pickle(out / f"{info['variant']}_episodes.pkl")
    (out / f"{info['variant']}.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="primary", choices=sorted(VARIANTS))
    ap.add_argument("--limit-days", type=int, default=0)
    a = ap.parse_args()
    p, m, info = build_panel(VARIANTS[a.variant], a.limit_days)
    save(p, m, info)
    print(json.dumps(info["funnel"], indent=1))
    if not p.empty:
        print(p.groupby(["k", "split"]).size().to_string())
