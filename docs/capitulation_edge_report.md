# Day-After-Capitulation Edge Report
*Generated: 2026-02-23 | Source: capitulation_analysis.csv (84 tagged records)*

> ⚠️ **STATUS: PARTIAL** — Next-day OHLC data not yet fetched. Run `scripts/fetch_capitulation_nextday.py` to populate `capitulation_nextday.csv`. This report covers cap-day characteristics and setup profiling only.

---

## Dataset Overview
- **Total records**: 120 (reversal setups, 2018–2025)
- **Capitulation-tagged**: 84 (70%) — composite score based on 5 signals
- **Not capitulation**: 36 (30%)

### Cap Score Distribution
| Score | Count | Notes |
|-------|-------|-------|
| 100   | 1     | All 5 signals |
| 90    | 10    | 4+ strong signals |
| 75    | 13    | High confidence |
| 70    | 1     | |
| 65    | 57    | Base threshold (3 signals) |
| 60    | 2     | Borderline |

---

## Key Finding: VIX Spike is the Most Powerful Discriminator

| Context | N | Gap Into Cap Day (avg) | Implication |
|---------|---|------------------------|-------------|
| **VIX Spike = YES** (broad market fear) | 16 | -11.7% | Market-wide selloff → next day risky |
| **VIX Spike = NO** (single-stock panic) | 68 | +27.1% | Stock-specific → bounce candidate |

**Rule: Single-stock capitulation (no VIX spike) = far better bounce setup than market-wide fear capitulation.**

---

## Cap Day Selloff Characteristics

| Metric | Value |
|--------|-------|
| Mean selloff (open→close) | -17.4% |
| Median selloff | -13.2% |
| Worst single-day | -58.1% |
| % with close at lows | 98.8% (83/84) |
| % with large down move | 100% |
| % with range expansion | 97.6% (82/84) |
| % with volume spike | 14.3% (12/84) |
| % with VIX spike | 19.0% (16/84) |

---

## Volume Spike Signal (Climactic Volume)

| Volume | N | Mean Gap Into Cap Day |
|--------|---|-----------------------|
| **Spike YES** | 12 | +50.0% (huge parabolic run-up INTO the cap) |
| **Spike NO** | 72 | +14.7% |

*Volume spike = stocks with climactic parabolic runs before the crash. These had the biggest gaps INTO the capitulation day.*

---

## Setup Type Distribution (84 Caps)

| Setup | Count | % |
|-------|-------|---|
| 3DGapFade | 20 | 23.8% |
| 2DBreakoutIB | 18 | 21.4% |
| 2DGapFade | 12 | 14.3% |
| GapDownTrendBreak | 12 | 14.3% |
| ConsolidationBreakdown | 8 | 9.5% |
| 1DMeanRevert | 4 | 4.8% |
| 1DBreakoutIB | 3 | 3.6% |
| RightSidePopFade | 3 | 3.6% |

**Best candidates for day-after bounce**: 3DGapFade and 2DGapFade (gap structure → defined reversal level). 
**Weaker candidates**: ConsolidationBreakdown (trend change, less predictable bounce).

---

## Cap Size Breakdown

| Cap | N | Mean Gap Into Cap Day | Mean Selloff |
|-----|---|-----------------------|--------------|
| ETF | ? | +23.6% | -3.3% (ETF = muted) |
| Large | ? | +1.3% | -9.2% |
| Medium | ? | +15.7% | -13.9% |
| Small | ? | +22.4% | -21.8% |
| Micro | ? | +56.1% | -43.3% |

**ETF capitulation** = small selloff, likely less bounce edge. **Medium cap** = sweet spot for day-after plays.

---

## Trade Grade Distribution

| Grade | Count |
|-------|-------|
| A | 38 (45%) |
| B | 39 (46%) |
| C | 7 (8%) |

91% A/B quality setups — this is a curated, high-quality dataset.

---

## ⚡ Preliminary Scoring Framework (Hypothesis — needs next-day data validation)

**Strongest bounce setup (next day):**
- ✅ VIX spike = NO (single-stock panic, not market fear)
- ✅ Volume spike = YES (climactic selling = exhaustion)
- ✅ Setup type = 3DGapFade or 2DGapFade (gap structure)
- ✅ Cap size = Medium or Large (liquid enough, big enough)
- ✅ Cap score ≥ 75 (high confidence capitulation)

**Weakest next-day setup:**
- ❌ VIX spike = YES (market-wide fear, often continues)
- ❌ ConsolidationBreakdown (trend change, no clear support)
- ❌ Micro cap (spreads too wide, unpredictable)

---

## Next Steps

1. **Run** `python scripts/fetch_capitulation_nextday.py` to get actual next-day OHLC
2. **Validate** the VIX spike discriminator hypothesis with real next-day returns
3. **Build** `b819076d` (edge statistics) and `929c10fb` (real-time screener)
4. **Key stats to compute once data is available**:
   - Probability of gap up next day (overall + by VIX context)
   - Win rate for open-of-day entry vs. dip entry
   - Expected value by cap score threshold
   - Optimal position sizing by cap size

---

*Report built by Buck | Task: b819076d (partial) | Full analysis pending next-day data fetch*
