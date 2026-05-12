"""
Crack Day Covering Rules — Adaptive decision tree for covering short positions
during parabolic crack days.

Based on analysis of 14 crack day case studies, classifies intraday patterns
by M1 size (ATRs), failed PBB count, and M2/M1 ratio, then applies
position-sizing rules at each significant held PBB.

Usage:
    from analyzers.crack_covering_rules import CoveringRules
    rules = CoveringRules()
    result = rules.simulate(moves, hod_price, lod_price, close_price)

    python analyzers/crack_covering_rules.py          # run validation on all 14 case studies
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CoverFill:
    """A single covering fill."""
    move_num: int              # Which move triggered this cover (0 = EOD)
    cover_pct: float           # % of original position covered (0.0-1.0)
    cover_price: float         # Price at which covered
    position_remaining: float  # % remaining after this fill (0.0-1.0)
    reason: str


@dataclass
class CoverResult:
    """Full simulation result for a crack day covering strategy."""
    fills: List[CoverFill] = field(default_factory=list)
    pattern_sequence: str = ""
    avg_cover_price: float = 0.0
    pct_captured: float = 0.0     # % of total crack captured (0-100)
    vs_lod_pct: float = 0.0       # vs optimal LOD cover (always <= 0)
    vs_naive_pct: float = 0.0     # vs covering 100% at M1 PBB


class CoveringRules:
    """
    Adaptive covering decision tree for crack days.

    The rules refine in real time as each move completes:

    AFTER M1 (first significant held PBB):
    ├── M1 >= 2.0 ATR
    │   ├── fail PBBs <= 1 → ONE_FLUSH_CLEAN  → Cover 75%
    │   └── fail PBBs >= 2 → ONE_FLUSH_STRONG → Cover 50%
    ├── M1 < 1.0 ATR       → PROBE            → Cover 0%
    └── M1 1.0–2.0 ATR     → DEVELOPING       → Cover 25%

    AFTER M2:
    ├── Was ONE_FLUSH_*     → Cover remaining
    ├── Was PROBE           → Cover 50%, trail rest
    ├── Was DEVELOPING
    │   ├── M2/M1 >= 1.0   → STAIRCASE        → Cover 33% more
    │   ├── M2 < 0.5 ATR   → DELAYED_FLUSH    → Cover 0% more (wait M3)
    │   └── else            → FADING           → Cover 50% more

    AFTER M3+:
    └── Cover 50% of whatever remains, trail the last portion.
        Final remainder covered at EOD.
    """

    def classify_after_m1(self, m1_atrs: float, m1_fail_pbbs: int
                          ) -> Tuple[str, float, str]:
        """
        Classify pattern and determine covering % after Move 1 completes.

        Returns: (pattern, cover_pct_of_original, reasoning)
        """
        if m1_atrs >= 2.0:
            if m1_fail_pbbs <= 1:
                return ("ONE_FLUSH_CLEAN", 0.75,
                        f"M1={m1_atrs:.1f} ATR, {m1_fail_pbbs} fail PBBs "
                        f"-> big clean flush, cover 75%")
            else:
                return ("ONE_FLUSH_STRONG", 0.50,
                        f"M1={m1_atrs:.1f} ATR, {m1_fail_pbbs} fail PBBs "
                        f"-> strong flush w/ resistance, cover 50%")
        elif m1_atrs < 1.0:
            return ("PROBE", 0.0,
                    f"M1={m1_atrs:.1f} ATR -> small probe, hold full position")
        else:
            return ("DEVELOPING", 0.25,
                    f"M1={m1_atrs:.1f} ATR -> developing crack, cover 25%")

    def classify_after_m2(self, pattern: str, m2_atrs: float,
                          m2_m1_ratio: float) -> Tuple[str, float, str]:
        """
        Refine pattern and determine additional covering after Move 2.

        For ONE_FLUSH_*: returns cover_pct = 1.0 meaning "cover all remaining".
        For others: returns the additional % of original position to cover.

        Returns: (refined_pattern, additional_cover_pct, reasoning)
        """
        if pattern.startswith("ONE_FLUSH"):
            return (pattern, 1.0,
                    f"Was {pattern} -> M2 confirms, cover remaining")

        if pattern == "PROBE":
            return ("PROBE", 0.50,
                    f"PROBE -> M2={m2_atrs:.1f} ATR, cover 50%")

        # DEVELOPING refinement
        if m2_m1_ratio >= 1.0:
            return ("STAIRCASE", 0.33,
                    f"DEVELOPING -> M2/M1={m2_m1_ratio:.2f}x >= 1.0, "
                    f"staircase pattern, cover 33% more")
        elif m2_atrs < 0.5:
            return ("DELAYED_FLUSH", 0.0,
                    f"DEVELOPING -> M2={m2_atrs:.1f} ATR < 0.5, "
                    f"delayed flush, hold for M3")
        else:
            return ("FADING", 0.50,
                    f"DEVELOPING -> M2/M1={m2_m1_ratio:.2f}x, "
                    f"fading momentum, cover 50% more")

    def classify_after_m3(self, remaining_pct: float) -> Tuple[float, str]:
        """
        After M3+: cover half of whatever remains, trail the rest.

        Returns: (cover_pct_of_original, reasoning)
        """
        cover = remaining_pct * 0.50
        return (cover,
                f"M3+: cover 50% of remaining "
                f"({remaining_pct*100:.0f}% -> {cover*100:.0f}%)")

    def simulate(self, moves: list, hod_price: float, lod_price: float,
                 close_price: float) -> CoverResult:
        """
        Walk through moves, apply the decision tree at each held PBB,
        track position remaining and cover prices.

        Args:
            moves: List of move dicts from CrackAnalysis.compute_move_analysis()
                   Each has: move_num, start_price, low_price, pbb_price,
                   size_atrs, failed_pbbs_during, etc.
            hod_price: High of day
            lod_price: Low of day
            close_price: Closing price (used for EOD covering)

        Returns: CoverResult with fills, pattern, and comparison metrics
        """
        result = CoverResult()
        position = 1.0  # Start fully short (1.0 = 100%)
        patterns = []

        if not moves:
            result.fills.append(CoverFill(
                0, 1.0, close_price, 0.0, "No moves detected, cover at close"))
            result.pattern_sequence = "NO_MOVES"
            result.avg_cover_price = close_price
            self._compute_comparisons(result, moves, hod_price, lod_price)
            return result

        # --- M1: first significant held PBB ---
        m1 = moves[0]
        m1_price = m1["pbb_price"] if m1["pbb_price"] > 0 else close_price
        m1_atrs = m1["size_atrs"]
        m1_fails = m1.get("failed_pbbs_during", 0)

        pattern, cover_pct, reasoning = self.classify_after_m1(m1_atrs, m1_fails)
        patterns.append(pattern)

        if cover_pct > 0:
            actual_cover = min(cover_pct, position)
            position -= actual_cover
            result.fills.append(CoverFill(
                1, actual_cover, m1_price, position, reasoning))

        # --- M2: second significant held PBB ---
        if len(moves) >= 2 and position > 0.001:
            m2 = moves[1]
            m2_price = m2["pbb_price"] if m2["pbb_price"] > 0 else close_price
            m2_atrs = m2["size_atrs"]
            m2_m1_ratio = m2_atrs / m1_atrs if m1_atrs > 0 else 0

            refined, add_cover, reasoning = self.classify_after_m2(
                pattern, m2_atrs, m2_m1_ratio)
            patterns.append(refined)

            if add_cover >= 1.0:
                # "cover remaining"
                actual_cover = position
            else:
                actual_cover = min(add_cover, position)

            if actual_cover > 0.001:
                position -= actual_cover
                result.fills.append(CoverFill(
                    2, actual_cover, m2_price, position, reasoning))

        # --- M3+: cover half of remaining at each subsequent held PBB ---
        for i in range(2, len(moves)):
            if position <= 0.001:
                break
            mv = moves[i]
            mv_price = mv["pbb_price"] if mv["pbb_price"] > 0 else close_price

            cover_amt, reasoning = self.classify_after_m3(position)
            if cover_amt > 0.001:
                actual_cover = min(cover_amt, position)
                position -= actual_cover
                result.fills.append(CoverFill(
                    mv["move_num"], actual_cover, mv_price, position, reasoning))

        # --- EOD: cover any remaining position at close ---
        if position > 0.001:
            result.fills.append(CoverFill(
                0, position, close_price, 0.0,
                f"EOD: cover final {position*100:.0f}% at close"))
            position = 0.0

        # --- Weighted average cover price ---
        result.pattern_sequence = " -> ".join(patterns)
        total_covered = sum(f.cover_pct for f in result.fills)
        if total_covered > 0:
            result.avg_cover_price = sum(
                f.cover_pct * f.cover_price for f in result.fills
            ) / total_covered

        self._compute_comparisons(result, moves, hod_price, lod_price)
        return result

    def _compute_comparisons(self, result: CoverResult, moves: list,
                             hod_price: float, lod_price: float):
        """Compute % captured, vs LOD (optimal), and vs naive (M1 PBB) metrics."""
        total_crack = hod_price - lod_price
        if total_crack <= 0:
            return

        # For shorts: profit = entry - cover price. Lower cover price = better.
        # pct_captured = (hod - avg_cover) / (hod - lod) * 100
        result.pct_captured = (
            (hod_price - result.avg_cover_price) / total_crack * 100
        )

        # vs LOD: always <= 0 (can't beat optimal)
        result.vs_lod_pct = result.pct_captured - 100.0

        # vs naive: covering 100% at M1 PBB price
        if moves:
            m1_pbb = moves[0]["pbb_price"]
            if m1_pbb > 0:
                naive_captured = (hod_price - m1_pbb) / total_crack * 100
            else:
                naive_captured = result.pct_captured
            result.vs_naive_pct = result.pct_captured - naive_captured


# ======================================================================
#  CLI — validate against all 14 case studies
# ======================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from analyzers.crack_analyzer import CrackAnalysis

    tickers = [
        ("GLD",  "2026-01-29"),
        ("MSTR", "2024-11-21"),
        ("SMCI", "2024-02-16"),
        ("NVDA", "2024-03-08"),
        ("BYND", "2025-10-22"),
        ("CLOV", "2021-06-09"),
        ("AMC",  "2021-01-28"),
        ("GME",  "2021-01-28"),
        ("BBBY", "2022-03-07"),
        ("MRNA", "2020-02-27"),
        ("FCEL", "2020-11-24"),
        ("COIN", "2022-08-04"),
        ("DJT",  "2024-10-30"),
        ("SPCE", "2021-07-12"),
    ]

    rules = CoveringRules()

    print(f"\n{'='*100}")
    print(f"  CRACK DAY COVERING RULES — VALIDATION")
    print(f"{'='*100}")
    print(f"  {'Ticker':<8} {'Pattern':<28} {'AvgCover':<12} {'%Captured':<11} "
          f"{'vsLOD':<9} {'vsNaive':<9} {'Fills'}")
    print(f"  {'-'*98}")

    results = []
    for ticker, date in tickers:
        try:
            a = CrackAnalysis(ticker, date)
            a.fetch_data()
            a.compute_vwap()
            a.detect_hod_lod()
            a.detect_legs()
            a.compute_metrics()
            a.compute_move_analysis()

            moves = a.metrics.get("move_analysis", {}).get("moves", [])
            close_price = a.rth_1m.iloc[-1]["close"]
            res = rules.simulate(
                moves, a.hod_price, a.lod_price, close_price)

            fill_str = " | ".join(
                f"M{f.move_num}:{f.cover_pct*100:.0f}%@${f.cover_price:.2f}"
                if f.move_num > 0 else
                f"EOD:{f.cover_pct*100:.0f}%@${f.cover_price:.2f}"
                for f in res.fills
            )

            print(f"  {ticker:<8} {res.pattern_sequence:<28} "
                  f"${res.avg_cover_price:<11.2f} {res.pct_captured:<10.1f}% "
                  f"{res.vs_lod_pct:<+8.1f}% {res.vs_naive_pct:<+8.1f}% "
                  f"{fill_str}")

            results.append(res)
        except Exception as e:
            print(f"  {ticker:<8} *** ERROR: {e}")

    # Averages
    if results:
        avg_cap = sum(r.pct_captured for r in results) / len(results)
        avg_lod = sum(r.vs_lod_pct for r in results) / len(results)
        avg_naive = sum(r.vs_naive_pct for r in results) / len(results)
        print(f"\n  {'AVERAGE':<8} {'':<28} {'':12} {avg_cap:<10.1f}% "
              f"{avg_lod:<+8.1f}% {avg_naive:<+8.1f}%")
        print(f"\n  {len(results)} tickers simulated successfully.")
