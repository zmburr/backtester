"""
Crack Day Analyzer — Study intraday anatomy of parabolic short crack days.

When a parabolic stock finally cracks, it typically makes:
  Leg 1 down -> Consolidation -> Leg 2 down (and sometimes Leg 3)

This module detects that structure, computes metrics for each phase,
and generates annotated charts to study covering behavior.

Usage:
    python analyzers/crack_analyzer.py GLD 2026-01-29
    python analyzers/crack_analyzer.py MSTR 2024-11-21
    python analyzers/crack_analyzer.py                   # run all case studies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_queries.polygon_queries import get_intraday, get_daily, get_atr

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --Case studies ────────────────────────────────────────────────────────
CRACK_DAYS = [
    {"ticker": "GLD",  "date": "2026-01-29", "cap": "ETF",    "notes": "Covered ~490 on 2m PBB, went to 470"},
    {"ticker": "MSTR", "date": "2024-11-21", "cap": "Large",  "notes": "All-day grind, 31.5% crack"},
    {"ticker": "SMCI", "date": "2024-02-16", "cap": "Medium", "notes": "Parabolic AI name crack"},
    {"ticker": "NVDA", "date": "2024-03-08", "cap": "Large",  "notes": "Post-GTC reversal"},
]


# =======================================================================
#  Core Analysis
# =======================================================================

class CrackAnalysis:
    """Full intraday analysis of a parabolic crack day."""

    def __init__(self, ticker: str, date: str, direction: str = "short"):
        self.ticker = ticker
        self.date = date
        self.direction = direction

        # Data
        self.bars_1m = None
        self.rth_1m = None        # regular-hours 1-min bars
        self.pm_1m = None         # premarket 1-min bars
        self.daily = None
        self.atr = None
        self.atr_dollars = None

        # Structure
        self.hod_price = None
        self.hod_time = None
        self.hod_pos = None       # integer position in rth_1m
        self.lod_price = None
        self.lod_time = None
        self.lod_pos = None

        # Phases detected: list of dicts
        self.phases = []

        # VWAP
        self.vwap = None

        # Computed metrics
        self.metrics = {}

    # --data fetching ───────────────────────────────────────────────
    def fetch_data(self):
        """Pull 1-min intraday bars, daily bar, and ATR."""
        self.bars_1m = get_intraday(self.ticker, self.date, 1, "minute")
        if self.bars_1m is None or self.bars_1m.empty:
            raise ValueError(f"No intraday data for {self.ticker} on {self.date}")

        self.rth_1m = self.bars_1m.between_time("09:30:00", "15:59:59").copy()
        self.pm_1m = self.bars_1m.between_time("04:00:00", "09:29:59").copy()

        self.daily = get_daily(self.ticker, self.date)
        self.atr = get_atr(self.ticker, self.date)

        # ATR in dollar terms (get_atr returns ATR as absolute $)
        self.atr_dollars = self.atr

    # --VWAP ────────────────────────────────────────────────────────
    def compute_vwap(self):
        df = self.rth_1m.copy()
        df["typical"] = (df["high"] + df["low"] + df["close"]) / 3
        df["cum_tp_vol"] = (df["typical"] * df["volume"]).cumsum()
        df["cum_vol"] = df["volume"].cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]
        self.vwap = df["vwap"]

    # --HOD / LOD detection ─────────────────────────────────────────
    def detect_hod_lod(self):
        self.hod_time = self.rth_1m["high"].idxmax()
        self.hod_price = self.rth_1m["high"].max()
        self.hod_pos = self.rth_1m.index.get_loc(self.hod_time)

        self.lod_time = self.rth_1m["low"].idxmin()
        self.lod_price = self.rth_1m["low"].min()
        self.lod_pos = self.rth_1m.index.get_loc(self.lod_time)

    # --Leg / consolidation detection ───────────────────────────────
    def detect_legs(self, consol_bars: int = 5, retrace_pct: float = 0.20,
                    min_leg_atrs: float = 0.5):
        """
        Walk forward from HOD and segment into legs and consolidations.

        A consolidation starts when price has not made a new low for
        *consol_bars* consecutive bars AND has retraced at least *retrace_pct*
        of the preceding leg.

        A new leg begins when the consolidation low is broken.

        This repeats, so we can detect Leg1->Consol->Leg2->Consol->Leg3 etc.
        """
        df = self.rth_1m
        post_hod = df.iloc[self.hod_pos:]
        if len(post_hod) < 3:
            return

        self.phases = []
        leg_num = 1
        i_start = 0                           # start of current leg (relative to post_hod)
        running_low = post_hod["high"].iloc[0] # initialize high so first bar sets it
        bars_no_new_low = 0

        i = 0
        while i < len(post_hod):
            low = post_hod["low"].iloc[i]

            if low < running_low:
                running_low = low
                bars_no_new_low = 0
            else:
                bars_no_new_low += 1

            # Check for consolidation start
            if bars_no_new_low >= consol_bars:
                # Verify retrace condition
                leg_top = post_hod["high"].iloc[i_start] if leg_num == 1 else post_hod.iloc[i_start:i+1]["high"].max()
                leg_drop = leg_top - running_low
                recent_high = post_hod.iloc[max(i_start, i - consol_bars):i + 1]["high"].max()
                retrace = recent_high - running_low

                if leg_drop > 0 and (retrace / leg_drop) >= retrace_pct:
                    # Record the leg that just ended
                    leg_end_i = i - bars_no_new_low
                    leg_bars = post_hod.iloc[i_start:leg_end_i + 1]
                    if len(leg_bars) > 0:
                        self._add_phase(
                            f"leg_down_{leg_num}", post_hod, i_start, leg_end_i
                        )

                    # Now find where consolidation ends (price breaks running_low)
                    consol_start_i = leg_end_i + 1
                    consol_end_i = consol_start_i
                    consol_high = running_low  # track consolidation high for retrace calc
                    for j in range(consol_start_i, len(post_hod)):
                        if post_hod["low"].iloc[j] < running_low:
                            consol_end_i = j - 1
                            break
                        consol_high = max(consol_high, post_hod["high"].iloc[j])
                        consol_end_i = j

                    consol_bars_slice = post_hod.iloc[consol_start_i:consol_end_i + 1]
                    if len(consol_bars_slice) > 0:
                        retrace_of_leg = (consol_bars_slice["high"].max() - running_low) / leg_drop if leg_drop > 0 else 0
                        self._add_phase(
                            "consolidation", post_hod, consol_start_i, consol_end_i,
                            extra={"retrace_pct": retrace_of_leg}
                        )

                    # Set up for next leg
                    leg_num += 1
                    i_start = consol_end_i + 1
                    i = i_start
                    bars_no_new_low = 0
                    # running_low stays the same — next leg must break it
                    continue

            i += 1

        # Remaining bars after last consolidation = final leg (or only leg)
        if i_start < len(post_hod):
            remaining = post_hod.iloc[i_start:]
            if len(remaining) > 0:
                self._add_phase(
                    f"leg_down_{leg_num}", post_hod, i_start, len(post_hod) - 1
                )

        # Merge small legs into adjacent consolidations
        if self.atr_dollars and min_leg_atrs > 0:
            self._merge_small_phases(min_leg_atrs)

    def _add_phase(self, phase_type: str, post_hod, start_i: int, end_i: int, extra: dict = None):
        """Helper to build a phase dict."""
        bars = post_hod.iloc[start_i:end_i + 1]
        if len(bars) == 0:
            return
        phase = {
            "type": phase_type,
            "start_time": bars.index[0],
            "end_time": bars.index[-1],
            "start_price": bars.iloc[0]["open"],
            "end_price": bars.iloc[-1]["close"],
            "high": bars["high"].max(),
            "low": bars["low"].min(),
            "duration_bars": len(bars),
            "total_volume": int(bars["volume"].sum()),
            "avg_bar_volume": int(bars["volume"].mean()),
        }
        if extra:
            phase.update(extra)
        self.phases.append(phase)

    def _merge_small_phases(self, min_leg_atrs: float):
        """Merge legs smaller than min_leg_atrs into adjacent consolidations."""
        if not self.phases or not self.atr_dollars:
            return

        merged = []
        for phase in self.phases:
            if "leg" in phase["type"]:
                leg_size = phase["high"] - phase["low"]
                leg_atrs = leg_size / self.atr_dollars
                if leg_atrs < min_leg_atrs and merged:
                    # Too small -- absorb into previous phase (extend its time/range)
                    prev = merged[-1]
                    prev["end_time"] = phase["end_time"]
                    prev["end_price"] = phase["end_price"]
                    prev["high"] = max(prev["high"], phase["high"])
                    prev["low"] = min(prev["low"], phase["low"])
                    prev["duration_bars"] += phase["duration_bars"]
                    prev["total_volume"] += phase["total_volume"]
                    prev["avg_bar_volume"] = prev["total_volume"] // prev["duration_bars"]
                    continue
            merged.append(phase)

        # Now merge consecutive consolidations that resulted from leg removal
        final = []
        for phase in merged:
            if final and phase["type"] == "consolidation" and final[-1]["type"] == "consolidation":
                prev = final[-1]
                prev["end_time"] = phase["end_time"]
                prev["end_price"] = phase["end_price"]
                prev["high"] = max(prev["high"], phase["high"])
                prev["low"] = min(prev["low"], phase["low"])
                prev["duration_bars"] += phase["duration_bars"]
                prev["total_volume"] += phase["total_volume"]
                prev["avg_bar_volume"] = prev["total_volume"] // prev["duration_bars"]
                # Keep the larger retrace
                prev["retrace_pct"] = max(prev.get("retrace_pct", 0), phase.get("retrace_pct", 0))
            else:
                final.append(phase)

        # Renumber legs
        leg_num = 1
        for phase in final:
            if "leg" in phase["type"]:
                phase["type"] = f"leg_down_{leg_num}"
                leg_num += 1

        self.phases = final

    # --Metrics ─────────────────────────────────────────────────────
    def compute_metrics(self):
        df = self.rth_1m
        m = {}

        # --Basic structure ──
        m["ticker"] = self.ticker
        m["date"] = self.date
        m["hod_price"] = self.hod_price
        m["hod_time"] = str(self.hod_time)
        m["lod_price"] = self.lod_price
        m["lod_time"] = str(self.lod_time)
        m["total_crack_pct"] = (self.hod_price - self.lod_price) / self.hod_price
        m["total_crack_dollars"] = self.hod_price - self.lod_price
        m["total_crack_atrs"] = (
            m["total_crack_dollars"] / self.atr_dollars if self.atr_dollars else None
        )
        m["hod_to_lod_bars"] = self.lod_pos - self.hod_pos
        m["hod_to_lod_minutes"] = m["hod_to_lod_bars"]  # 1-min bars

        # Close vs LOD — did it bounce into close?
        close_price = df.iloc[-1]["close"]
        m["close_price"] = close_price
        m["close_vs_lod_pct"] = (close_price - self.lod_price) / self.lod_price if self.lod_price else 0
        m["close_vs_hod_pct"] = (self.hod_price - close_price) / self.hod_price

        # --Volume ──
        total_vol = int(df["volume"].sum())
        pre_hod = df.iloc[: self.hod_pos + 1]
        post_hod = df.iloc[self.hod_pos:]

        m["total_day_volume"] = total_vol
        m["pre_hod_volume"] = int(pre_hod["volume"].sum())
        m["post_hod_volume"] = int(post_hod["volume"].sum())
        m["post_hod_vol_pct"] = post_hod["volume"].sum() / total_vol if total_vol else 0

        avg_bar_vol = df["volume"].mean()
        hod_bar_vol = df.iloc[self.hod_pos]["volume"]
        m["hod_bar_volume"] = int(hod_bar_vol)
        m["hod_bar_vol_ratio"] = hod_bar_vol / avg_bar_vol if avg_bar_vol else 0

        # Peak 5-bar volume (rolling 5-bar sum, max value post-HOD)
        post_vol = post_hod["volume"]
        if len(post_vol) >= 5:
            rolling5 = post_vol.rolling(5).sum()
            m["peak_5bar_volume"] = int(rolling5.max())
            m["peak_5bar_time"] = str(rolling5.idxmax())
        else:
            m["peak_5bar_volume"] = int(post_vol.sum())

        # --Momentum / acceleration ──
        post_hod_closes = post_hod["close"]
        bar_changes = post_hod_closes.diff()

        if m["hod_to_lod_minutes"] > 0:
            m["crack_rate_pct_per_min"] = m["total_crack_pct"] / m["hod_to_lod_minutes"]
            m["crack_rate_dollars_per_min"] = m["total_crack_dollars"] / m["hod_to_lod_minutes"]
        else:
            m["crack_rate_pct_per_min"] = 0
            m["crack_rate_dollars_per_min"] = 0

        m["max_single_bar_drop"] = float(bar_changes.min()) if len(bar_changes.dropna()) > 0 else 0
        m["max_single_bar_drop_pct"] = m["max_single_bar_drop"] / self.hod_price if self.hod_price else 0
        m["avg_bar_change_during_crack"] = float(bar_changes.mean()) if len(bar_changes.dropna()) > 0 else 0

        # Consecutive red bars from HOD
        consecutive_red = 0
        for ch in bar_changes.dropna():
            if ch < 0:
                consecutive_red += 1
            else:
                break
        m["consecutive_red_from_hod"] = consecutive_red

        # Max consecutive red bars anywhere in crack
        max_consec_red = 0
        current_streak = 0
        for ch in bar_changes.dropna():
            if ch < 0:
                current_streak += 1
                max_consec_red = max(max_consec_red, current_streak)
            else:
                current_streak = 0
        m["max_consecutive_red_bars"] = max_consec_red

        # --VWAP context ──
        if self.vwap is not None and len(self.vwap) > 0:
            vwap_at_hod = self.vwap.iloc[min(self.hod_pos, len(self.vwap) - 1)]
            vwap_at_lod = self.vwap.iloc[min(self.lod_pos, len(self.vwap) - 1)]
            m["hod_vs_vwap_pct"] = (self.hod_price - vwap_at_hod) / vwap_at_hod
            m["lod_vs_vwap_pct"] = (self.lod_price - vwap_at_lod) / vwap_at_lod
            m["vwap_at_close"] = float(self.vwap.iloc[-1])

        # --Phase-specific metrics ──
        for phase in self.phases:
            p = phase["type"]
            phase_size = phase["high"] - phase["low"]
            m[f"{p}_size_dollars"] = phase_size
            m[f"{p}_size_pct"] = phase_size / phase["high"] if phase["high"] else 0
            m[f"{p}_size_atrs"] = phase_size / self.atr_dollars if self.atr_dollars else 0
            m[f"{p}_duration_bars"] = phase["duration_bars"]
            m[f"{p}_volume"] = phase["total_volume"]
            m[f"{p}_avg_bar_vol"] = phase["avg_bar_volume"]

        # --Leg ratios ──
        legs = [p for p in self.phases if "leg" in p["type"]]
        consols = [p for p in self.phases if p["type"] == "consolidation"]

        if len(legs) >= 2:
            l1_size = legs[0]["high"] - legs[0]["low"]
            l2_size = legs[1]["high"] - legs[1]["low"]
            m["leg2_to_leg1_size_ratio"] = l2_size / l1_size if l1_size else 0
            m["leg1_pct_of_total"] = l1_size / m["total_crack_dollars"] if m["total_crack_dollars"] else 0
            m["leg2_pct_of_total"] = l2_size / m["total_crack_dollars"] if m["total_crack_dollars"] else 0
            m["leg2_vol_to_leg1_vol"] = legs[1]["total_volume"] / legs[0]["total_volume"] if legs[0]["total_volume"] else 0
            m["leg2_duration_to_leg1"] = legs[1]["duration_bars"] / legs[0]["duration_bars"] if legs[0]["duration_bars"] else 0

        if len(consols) >= 1:
            m["consol_retrace_pct"] = consols[0].get("retrace_pct", 0)
            if len(legs) >= 1:
                m["consol_vol_vs_leg1"] = consols[0]["total_volume"] / legs[0]["total_volume"] if legs[0]["total_volume"] else 0

        # --Prior bar break analysis (on 2-min bars) ──
        bars_2m = get_intraday(self.ticker, self.date, 2, "minute")
        if bars_2m is not None and not bars_2m.empty:
            rth_2m = bars_2m.between_time("09:30:00", "15:59:59")
            m["prior_bar_breaks"] = self._analyze_prior_bar_breaks(rth_2m)
            m["num_prior_bar_breaks"] = len(m["prior_bar_breaks"])
            m["num_failed_pbb"] = sum(1 for pb in m["prior_bar_breaks"] if not pb["held"])

        # --ATR-level hits ──
        if self.atr_dollars:
            for mult in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                level = self.hod_price - (mult * self.atr_dollars)
                hit = self.lod_price <= level
                m[f"hit_{mult}x_atr"] = hit
                if hit:
                    # Find when it was first hit
                    post = self.rth_1m.iloc[self.hod_pos:]
                    hits = post[post["low"] <= level]
                    if len(hits) > 0:
                        m[f"time_to_{mult}x_atr_bars"] = self.rth_1m.index.get_loc(hits.index[0]) - self.hod_pos

        self.metrics = m
        return m

    def _analyze_prior_bar_breaks(self, bars_df):
        """
        On 2-min bars post-HOD, find every instance where a bar's high
        breaks the prior bar's high (potential cover signal for shorts).
        Then check if the break held or failed (price went below prior bar low within 3 bars).
        """
        hod_time = self.hod_time
        # Find HOD position in this timeframe
        hod_pos = 0
        for idx, row in bars_df.iterrows():
            if idx >= hod_time:
                break
            hod_pos += 1

        post = bars_df.iloc[hod_pos:]
        breaks = []

        for i in range(2, len(post)):
            curr = post.iloc[i]
            prior = post.iloc[i - 1]

            if curr["high"] > prior["high"]:
                # Prior bar high was broken — cover signal
                held = True
                for j in range(i + 1, min(i + 4, len(post))):
                    if post.iloc[j]["low"] < prior["low"]:
                        held = False
                        break

                breaks.append({
                    "time": str(post.index[i]),
                    "price": float(prior["high"]),
                    "held": held,
                    "bars_from_hod": i,
                    "pct_from_hod": (self.hod_price - prior["high"]) / self.hod_price,
                    "pct_from_lod": (prior["high"] - self.lod_price) / self.lod_price if self.lod_price else 0,
                })

        return breaks

    def compute_move_analysis(self, min_move_atrs: float = 0.5):
        """
        The key analysis: structured moves between SIGNIFICANT held PBBs on 2-min bars.

        A held PBB only counts as a move boundary if the drop from the move
        start is >= min_move_atrs ATRs. This filters out opening noise where
        tiny bounces technically "hold" but aren't real pauses.

        Move 1 = HOD down to the low at the first significant held PBB
        Move 2 = from bounce high after first held PBB, down to next significant held PBB
        ...and so on.

        The 'failed_pbbs_during' count tells you how many bounce attempts failed
        during the move — the real-time momentum signal.
        """
        bars_2m = get_intraday(self.ticker, self.date, 2, "minute")
        if bars_2m is None or bars_2m.empty:
            return {}

        rth_2m = bars_2m.between_time("09:30:00", "15:59:59")

        # Find HOD bar position in 2-min data
        hod_pos = 0
        for idx in rth_2m.index:
            if idx >= self.hod_time:
                break
            hod_pos += 1

        post = rth_2m.iloc[hod_pos:]
        if len(post) < 4:
            return {}

        min_move_dollars = min_move_atrs * self.atr_dollars if self.atr_dollars else 0

        # Find all PBBs and whether they held
        pbbs = []
        for i in range(2, len(post)):
            curr = post.iloc[i]
            prior = post.iloc[i - 1]

            if curr["high"] > prior["high"]:
                held = True
                for j in range(i + 1, min(i + 4, len(post))):
                    if post.iloc[j]["low"] < prior["low"]:
                        held = False
                        break
                pbbs.append({
                    "bar_idx": i,
                    "time": post.index[i],
                    "pbb_price": float(prior["high"]),
                    "held": held,
                })

        if not pbbs:
            return {}

        # Build moves between SIGNIFICANT held PBBs
        # A held PBB is only a boundary if the move from start is >= min_move_atrs
        moves = []
        move_start_price = self.hod_price
        move_start_idx = 0
        running_low = self.hod_price
        skipped_held = 0  # held PBBs skipped because move was too small

        for pbb in pbbs:
            hi = pbb["bar_idx"]

            # Track running low across all bars up to this PBB
            segment = post.iloc[move_start_idx:hi + 1]
            if len(segment) > 0:
                seg_low = segment["low"].min()
                if seg_low < running_low:
                    running_low = seg_low

            if not pbb["held"]:
                continue  # failed PBBs don't end moves, just count them

            # This PBB held — but is the move big enough to matter?
            move_size = move_start_price - running_low
            if move_size < min_move_dollars:
                skipped_held += 1
                continue  # noise — skip this held PBB

            # Significant move boundary found
            low_time = segment["low"].idxmin() if len(segment) > 0 else post.index[move_start_idx]
            bounce_high = pbb["pbb_price"]
            move_num = len(moves) + 1

            # Count all failed PBBs + skipped held PBBs during this move
            failed_in_move = sum(1 for p in pbbs
                                if p["bar_idx"] >= move_start_idx
                                and p["bar_idx"] < hi
                                and not p["held"])

            move = {
                "move_num": move_num,
                "start_price": move_start_price,
                "low_price": float(running_low),
                "low_time": str(low_time),
                "pbb_time": str(pbb["time"]),
                "pbb_price": float(bounce_high),
                "size_dollars": move_size,
                "size_pct": move_size / move_start_price if move_start_price else 0,
                "size_atrs": move_size / self.atr_dollars if self.atr_dollars else 0,
                "bars": hi - move_start_idx,
                "volume": int(post.iloc[move_start_idx:hi + 1]["volume"].sum()),
                "failed_pbbs_during": failed_in_move,
                "skipped_held_pbbs": skipped_held,
            }
            moves.append(move)

            # Reset for next move
            move_start_price = bounce_high
            move_start_idx = hi + 1
            running_low = bounce_high
            skipped_held = 0

            if move_num >= 5:
                break

        # Final move: from last boundary to LOD (if significant)
        if move_start_idx < len(post):
            remaining = post.iloc[move_start_idx:]
            remaining_low = remaining["low"].min()
            move_size = move_start_price - remaining_low
            if move_size >= min_move_dollars:
                failed_remaining = sum(1 for p in pbbs
                                       if p["bar_idx"] >= move_start_idx
                                       and not p["held"])
                moves.append({
                    "move_num": len(moves) + 1,
                    "start_price": move_start_price,
                    "low_price": float(remaining_low),
                    "low_time": str(remaining["low"].idxmin()),
                    "pbb_time": "EOD",
                    "pbb_price": 0,
                    "size_dollars": move_size,
                    "size_pct": move_size / move_start_price if move_start_price else 0,
                    "size_atrs": move_size / self.atr_dollars if self.atr_dollars else 0,
                    "bars": len(remaining),
                    "volume": int(remaining["volume"].sum()),
                    "failed_pbbs_during": failed_remaining,
                    "skipped_held_pbbs": skipped_held,
                })

        # Compute ratios
        result = {"moves": moves}
        if len(moves) >= 1:
            result["move1_atrs"] = moves[0]["size_atrs"]
        if len(moves) >= 2:
            result["move2_atrs"] = moves[1]["size_atrs"]
            result["move2_to_move1_ratio"] = (
                moves[1]["size_atrs"] / moves[0]["size_atrs"]
                if moves[0]["size_atrs"] else 0
            )
        if len(moves) >= 3:
            result["move3_atrs"] = moves[2]["size_atrs"]
            result["move3_to_move1_ratio"] = (
                moves[2]["size_atrs"] / moves[0]["size_atrs"]
                if moves[0]["size_atrs"] else 0
            )

        self.metrics["move_analysis"] = result
        return result

    # --Charting ────────────────────────────────────────────────────
    def create_chart(self, bar_size: int = 2):
        """Annotated Plotly chart with phase shading, VWAP, and ATR levels."""
        chart_bars = get_intraday(self.ticker, self.date, bar_size, "minute")
        chart_bars = chart_bars.between_time("09:30:00", "15:59:59")

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=[
                f"{self.ticker} Crack Day — {self.date} ({bar_size}-min bars)",
                "Volume",
                "VWAP Distance %",
            ],
        )

        # --Row 1: Candlestick ──
        fig.add_trace(
            go.Candlestick(
                x=chart_bars.index, open=chart_bars["open"],
                high=chart_bars["high"], low=chart_bars["low"],
                close=chart_bars["close"], name="Price",
            ),
            row=1, col=1,
        )

        # VWAP overlay
        if self.vwap is not None:
            fig.add_trace(
                go.Scatter(
                    x=self.vwap.index, y=self.vwap.values,
                    mode="lines", name="VWAP",
                    line=dict(color="dodgerblue", width=1.5, dash="dash"),
                ),
                row=1, col=1,
            )

        # ATR level lines from HOD
        if self.atr_dollars:
            for mult, color, dash in [
                (1.0, "orange", "dot"), (1.5, "orangered", "dot"),
                (2.0, "red", "dot"), (2.5, "darkred", "dot"),
                (3.0, "purple", "dot"),
            ]:
                level = self.hod_price - (mult * self.atr_dollars)
                fig.add_hline(
                    y=level, line_dash=dash, line_color=color,
                    annotation_text=f"{mult}x ATR (${level:.2f})",
                    annotation_position="bottom right",
                    annotation_font_size=9,
                    row=1, col=1,
                )

        # HOD / LOD annotations
        fig.add_annotation(
            x=self.hod_time, y=self.hod_price,
            text=f"HOD ${self.hod_price:.2f}", showarrow=True, arrowhead=2,
            bgcolor="red", font=dict(color="white"), row=1, col=1,
        )
        fig.add_annotation(
            x=self.lod_time, y=self.lod_price,
            text=f"LOD ${self.lod_price:.2f}", showarrow=True, arrowhead=2,
            bgcolor="green", font=dict(color="white"), row=1, col=1,
        )

        # Phase shading -- consolidation gets horizontal band, legs get vertical shading
        leg_colors = ["rgba(255,80,80,0.12)", "rgba(255,40,40,0.18)",
                      "rgba(200,0,0,0.22)", "rgba(180,0,0,0.25)"]
        leg_idx = 0
        for phase in self.phases:
            if "leg" in phase["type"]:
                fc = leg_colors[min(leg_idx, len(leg_colors) - 1)]
                leg_idx += 1
                size_pct = (phase["high"] - phase["low"]) / phase["high"] * 100 if phase["high"] else 0
                atrs = (phase["high"] - phase["low"]) / self.atr_dollars if self.atr_dollars else 0
                fig.add_vrect(
                    x0=phase["start_time"], x1=phase["end_time"],
                    fillcolor=fc, line_width=0,
                    annotation_text=f"{phase['type'].replace('_', ' ').title()}<br>{atrs:.1f}x ATR",
                    annotation_position="top left",
                    annotation_font_size=9,
                    row=1, col=1,
                )
            elif phase["type"] == "consolidation":
                # Horizontal band showing consolidation range
                retrace = phase.get("retrace_pct", 0)
                fig.add_hrect(
                    y0=phase["low"], y1=phase["high"],
                    fillcolor="rgba(80,80,255,0.10)", line_width=1,
                    line_color="rgba(80,80,255,0.3)",
                    row=1, col=1,
                )
                # Label at the right edge
                mid_price = (phase["high"] + phase["low"]) / 2
                fig.add_annotation(
                    x=phase["end_time"], y=mid_price,
                    text=f"Consol {retrace*100:.0f}% retrace<br>{phase['duration_bars']} bars",
                    showarrow=False, bgcolor="rgba(80,80,255,0.7)",
                    font=dict(color="white", size=9),
                    row=1, col=1,
                )

        # Move analysis annotations on chart
        ma = self.metrics.get("move_analysis", {})
        move_colors = ["#FF6600", "#CC0000", "#660066", "#003366"]
        for mv in ma.get("moves", [])[:4]:
            i = mv["move_num"] - 1
            color = move_colors[min(i, len(move_colors) - 1)]
            fails = mv.get("failed_pbbs_during", 0)
            # Arrow at the low of the move
            fig.add_annotation(
                x=pd.Timestamp(mv["low_time"]), y=mv["low_price"],
                text=f"M{mv['move_num']}: {mv['size_atrs']:.1f}x ATR ({fails} fail PBBs)",
                showarrow=True, arrowhead=2, arrowcolor=color,
                bgcolor=color, font=dict(color="white", size=10),
                ay=-40 if i % 2 == 0 else 40,
                row=1, col=1,
            )
            # Held PBB marker (the boundary that ended this move)
            if mv["pbb_price"] > 0:  # skip EOD
                fig.add_annotation(
                    x=pd.Timestamp(mv["pbb_time"]), y=mv["pbb_price"],
                    text=f"PBB HELD",
                    showarrow=True, arrowhead=1,
                    bgcolor="lime",
                    font=dict(color="black", size=9),
                    ay=-25,
                row=1, col=1,
            )

        # Prior bar break markers (from metrics)
        pbb = self.metrics.get("prior_bar_breaks", [])
        for pb in pbb:
            color = "lime" if pb["held"] else "yellow"
            symbol = "triangle-up" if pb["held"] else "x"
            fig.add_trace(
                go.Scatter(
                    x=[pd.Timestamp(pb["time"])], y=[pb["price"]],
                    mode="markers", marker=dict(color=color, size=7, symbol=symbol,
                                                line=dict(width=0.5, color="black"), opacity=0.6),
                    name=f"PBB {'held' if pb['held'] else 'fail'}",
                    showlegend=False,
                    hovertext=f"PBB @ ${pb['price']:.2f} ({pb['pct_from_hod']*100:.1f}% from HOD) - {'HELD' if pb['held'] else 'FAILED'}",
                ),
                row=1, col=1,
            )

        # --Row 2: Volume bars colored by phase ──
        vol_colors = self._color_bars_by_phase(chart_bars)
        fig.add_trace(
            go.Bar(x=chart_bars.index, y=chart_bars["volume"], marker_color=vol_colors, name="Volume", showlegend=False),
            row=2, col=1,
        )

        # --Row 3: VWAP distance % ──
        if self.vwap is not None:
            close_s = self.rth_1m["close"]
            vwap_dist = ((close_s - self.vwap) / self.vwap) * 100
            colors = ["red" if v < 0 else "green" for v in vwap_dist]
            fig.add_trace(
                go.Bar(x=vwap_dist.index, y=vwap_dist.values, marker_color=colors, name="VWAP Dist %", showlegend=False),
                row=3, col=1,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

        fig.update_layout(
            height=950, width=1400,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(rangeslider_visible=False)

        return fig

    def _color_bars_by_phase(self, chart_bars):
        """Assign a color to each chart bar based on which phase it falls in."""
        colors = []
        for idx in chart_bars.index:
            matched = False
            for phase in self.phases:
                if phase["start_time"] <= idx <= phase["end_time"]:
                    if "leg" in phase["type"]:
                        colors.append("crimson")
                    else:
                        colors.append("royalblue")
                    matched = True
                    break
            if not matched:
                colors.append("gray")
        return colors

    # --Pretty print ────────────────────────────────────────────────
    def print_summary(self):
        m = self.metrics
        print(f"\n{'='*70}")
        print(f"  CRACK ANATOMY: {self.ticker} on {self.date}")
        print(f"{'='*70}")

        print(f"\n  HOD: ${m['hod_price']:.2f}  at  {m['hod_time']}")
        print(f"  LOD: ${m['lod_price']:.2f}  at  {m['lod_time']}")
        print(f"  Total Crack: ${m['total_crack_dollars']:.2f}  ({m['total_crack_pct']*100:.1f}%)")
        if m.get("total_crack_atrs"):
            print(f"  In ATRs: {m['total_crack_atrs']:.1f}x")
        print(f"  Duration: {m['hod_to_lod_minutes']} min")
        print(f"  Close: ${m['close_price']:.2f}  ({m['close_vs_hod_pct']*100:.1f}% below HOD, {m['close_vs_lod_pct']*100:.1f}% above LOD)")

        print(f"\n  -- MOMENTUM --")
        print(f"  Rate: {m.get('crack_rate_pct_per_min',0)*100:.3f}%/min  |  ${m.get('crack_rate_dollars_per_min',0):.2f}/min")
        print(f"  Consec red from HOD: {m.get('consecutive_red_from_hod',0)}  |  Max streak: {m.get('max_consecutive_red_bars',0)}")
        print(f"  Biggest single bar: ${abs(m.get('max_single_bar_drop',0)):.2f}  ({abs(m.get('max_single_bar_drop_pct',0))*100:.2f}%)")

        print(f"\n  --VOLUME --")
        print(f"  Day total: {m.get('total_day_volume',0):,.0f}")
        print(f"  Post-HOD: {m.get('post_hod_vol_pct',0)*100:.0f}% of day")
        print(f"  HOD bar: {m.get('hod_bar_vol_ratio',0):.1f}x avg bar")

        print(f"\n  --PHASES --")
        for phase in self.phases:
            ptype = phase["type"].replace("_", " ").title()
            size_d = phase["high"] - phase["low"]
            size_pct = size_d / phase["high"] * 100 if phase["high"] else 0
            atrs = size_d / self.atr_dollars if self.atr_dollars else 0
            print(f"  {ptype}:")
            print(f"    ${size_d:.2f} ({size_pct:.1f}%)  |  {atrs:.1f}x ATR  |  {phase['duration_bars']} bars")
            print(f"    Volume: {phase['total_volume']:,} (avg/bar {phase['avg_bar_volume']:,})")
            if "retrace_pct" in phase:
                print(f"    Retrace of prior leg: {phase['retrace_pct']*100:.0f}%")

        # Leg ratios
        if m.get("leg2_to_leg1_size_ratio") is not None:
            print(f"\n  --LEG RATIOS --")
            print(f"  Leg2 / Leg1 size: {m['leg2_to_leg1_size_ratio']:.2f}x")
            print(f"  Leg2 / Leg1 volume: {m.get('leg2_vol_to_leg1_vol',0):.2f}x")
            print(f"  Leg2 / Leg1 duration: {m.get('leg2_duration_to_leg1',0):.2f}x")
            print(f"  Leg1 = {m.get('leg1_pct_of_total',0)*100:.0f}% of total crack")
            print(f"  Leg2 = {m.get('leg2_pct_of_total',0)*100:.0f}% of total crack")

        if m.get("consol_retrace_pct") is not None:
            print(f"\n  --CONSOLIDATION --")
            print(f"  Retrace of Leg 1: {m['consol_retrace_pct']*100:.0f}%")
            print(f"  Vol vs Leg 1: {m.get('consol_vol_vs_leg1',0):.2f}x")

        # ATR hits
        print(f"\n  --ATR LEVELS HIT --")
        for mult in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            hit = m.get(f"hit_{mult}x_atr", False)
            bars_key = f"time_to_{mult}x_atr_bars"
            bars_to = m.get(bars_key, "-")
            level = self.hod_price - (mult * self.atr_dollars) if self.atr_dollars else 0
            mark = "Y" if hit else " "
            print(f"  [{mark}] {mult}x ATR = ${level:.2f}   {'(' + str(bars_to) + ' bars)' if hit else ''}")

        # Move analysis (the key output)
        ma = m.get("move_analysis", {})
        moves = ma.get("moves", [])
        if moves:
            print(f"\n  ** MOVE ANALYSIS (moves between held 2m PBBs) **")
            print(f"  {'Move':<8} {'ATRs':<8} {'Size $':<10} {'%':<8} {'Bars':<6} {'Failed PBBs':<13} {'Held PBB @':<12}")
            print(f"  {'-'*65}")
            for mv in moves:
                pbb_str = f"${mv['pbb_price']:.2f}" if mv["pbb_price"] > 0 else "EOD"
                print(f"  Move {mv['move_num']:<3} {mv['size_atrs']:<8.2f} ${mv['size_dollars']:<9.2f} {mv['size_pct']*100:<7.1f}% {mv['bars']:<6} {mv.get('failed_pbbs_during',0):<13} {pbb_str}")
            print()
            if ma.get("move2_to_move1_ratio") is not None:
                print(f"  >>> Move1: {ma['move1_atrs']:.2f}x ATR  |  Move2: {ma['move2_atrs']:.2f}x ATR  |  Ratio: {ma['move2_to_move1_ratio']:.2f}x")
                if ma.get("move3_atrs") is not None:
                    print(f"  >>> Move3: {ma['move3_atrs']:.2f}x ATR  |  M3/M1 ratio: {ma['move3_to_move1_ratio']:.2f}x")

        # Prior bar breaks
        pbb = m.get("prior_bar_breaks", [])
        print(f"\n  --2-MIN PRIOR BAR BREAKS --")
        print(f"  Total: {m.get('num_prior_bar_breaks',0)}  |  Failed: {m.get('num_failed_pbb',0)}")
        for pb in pbb[:15]:  # show first 15 only
            status = "HELD" if pb["held"] else "FAILED"
            print(f"    {pb['time'][:19]}  ${pb['price']:.2f}  ({pb['pct_from_hod']*100:.1f}% off HOD)  -> {status}")
        if len(pbb) > 15:
            print(f"    ... ({len(pbb) - 15} more)")

        print(f"{'='*70}\n")

    # --Main runner ─────────────────────────────────────────────────
    def run(self, consol_bars: int = 5, retrace_pct: float = 0.20,
            min_leg_atrs: float = 0.5, bar_size: int = 2, show: bool = True):
        """Full pipeline: fetch -> analyze -> print -> chart."""
        print(f"\n>>> Analyzing {self.ticker} on {self.date} ...")
        self.fetch_data()
        self.compute_vwap()
        self.detect_hod_lod()
        self.detect_legs(consol_bars=consol_bars, retrace_pct=retrace_pct, min_leg_atrs=min_leg_atrs)
        self.compute_metrics()
        self.compute_move_analysis()
        self.print_summary()
        fig = self.create_chart(bar_size=bar_size)
        if show:
            fig.show()
        return fig


# =======================================================================
#  CLI
# =======================================================================

def run_all(bar_size=2, show=True):
    """Analyze all case studies in CRACK_DAYS."""
    results = []
    for case in CRACK_DAYS:
        a = CrackAnalysis(case["ticker"], case["date"])
        a.run(bar_size=bar_size, show=show)
        results.append(a)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parabolic Crack Day Analyzer")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol (e.g. GLD)")
    parser.add_argument("date", nargs="?", help="Date YYYY-MM-DD (e.g. 2026-01-29)")
    parser.add_argument("--bar-size", type=int, default=2, help="Chart candle size in minutes (default 2)")
    parser.add_argument("--consol-bars", type=int, default=5, help="Min bars w/o new low to detect consolidation")
    parser.add_argument("--retrace-pct", type=float, default=0.20, help="Min retrace of prior leg for consolidation")
    parser.add_argument("--no-show", action="store_true", help="Don't open chart in browser")
    args = parser.parse_args()

    if args.ticker and args.date:
        analysis = CrackAnalysis(args.ticker, args.date)
        analysis.run(
            consol_bars=args.consol_bars,
            retrace_pct=args.retrace_pct,
            bar_size=args.bar_size,
            show=not args.no_show,
        )
    else:
        run_all(bar_size=args.bar_size, show=not args.no_show)
