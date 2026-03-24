# Breakout Setup Criteria Specification

**Purpose**: Define measurable criteria for valid breakout setups to enable systematic identification, screening, and trading of breakout opportunities.

**Status**: Draft v1.0 — Pending validation against historical data

---

## What Constitutes a Breakout

A breakout setup occurs when price moves decisively above resistance (long) or below support (short), typically accompanied by volume expansion and momentum. Valid breakouts should have follow-through potential, not just be false breaks.

---

## Core Criteria

### 1. Consolidation Period (Required)
**Definition**: Price trading in a defined range before the breakout attempt.

| Attribute | Threshold | Notes |
|-----------|-----------|-------|
| Minimum consolidation days | 5 days | Can be consecutive or within 10-day window |
| Maximum consolidation days | 30 days | Longer = more significant but harder to time |
| Range compression | < 2x ATR | Tight consolidation preferred |
| Price action | Higher lows (bullish) or lower highs (bearish) | Shows accumulation/distribution |

**Measurement**: 
- Calculate daily range (High - Low) for consolidation period
- Average range should be < 2x the 14-day ATR
- Visual check: price oscillating between two clear boundaries

---

### 2. Volume Profile (Required)
**Definition**: Volume behavior during consolidation and on breakout day.

| Attribute | Threshold | Notes |
|-----------|-----------|-------|
| Pre-breakout volume trend | Declining or stable | Dry-up before breakout preferred |
| Breakout day RVOL | >= 1.5x | Must have volume expansion |
| Volume pattern | Climax or sustained | Single spike okay if > 2x RVOL |
| 5-day avg volume | >= 500K | Liquidity requirement |

**Measurement**:
- RVOL = Volume on breakout day / 20-day average volume
- Pre-breakout: Volume in last 3 days of consolidation vs first 3 days

---

### 3. Breakout Day Characteristics (Required)
**Definition**: The specific price action on the day of the breakout.

| Attribute | Long Setup | Short Setup | Notes |
|-----------|------------|-------------|-------|
| Gap | Optional, < 3% | Optional, < 3% | Large gaps = higher failure risk |
| Open vs prior close | Above (preferred) | Below (preferred) | Clean break, not gap recovery |
| Breakout magnitude | > 2% above resistance | > 2% below support | Needs to be decisive |
| Close position | In upper 50% of range | In lower 50% of range | Shows sustained pressure |
| Close vs open | Green (close > open) | Red (close < open) | Momentum confirmation |

**Measurement**:
- Breakout magnitude % = (Breakout price - Resistance level) / Resistance level
- Prior resistance = High of consolidation period (at least 2 touches)

---

### 4. Moving Average Context (Required)
**Definition**: Relationship to key moving averages.

| Attribute | Long Setup | Short Setup |
|-----------|------------|-------------|
| Price vs 9 EMA | Above or crossing up | Below or crossing down |
| Price vs 20 SMA | Above or near | Below or near |
| 9 EMA vs 20 SMA | 9 EMA rising, above 20 SMA | 9 EMA falling, below 20 SMA |
| 50 SMA trend | Rising or flat | Falling or flat |

**Context Rules**:
- **Best**: Breakout in direction of MA trend (trend-following)
- **Okay**: Breakout against MA but with strong volume (counter-trend)
- **Avoid**: Breakout into major MA resistance without catalyst

---

### 5. Market Context (Required)
**Definition**: Broader market conditions during breakout.

| Attribute | Ideal | Avoid |
|-----------|-------|-------|
| SPY trend | Aligned with breakout direction | Opposite direction |
| Sector trend | Leading or strong | Weak relative to market |
| VIX level | < 25 (stable) | > 35 (choppy, false breaks likely) |
| Market breadth | > 50% stocks above 20 MA | < 30% (weak underlying) |

---

## Scoring System (GO/CAUTION/NO-GO)

### Binary Criteria (Must all pass for GO)
| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Consolidation >= 5 days, range < 2x ATR | Required |
| 2 | Breakout day RVOL >= 1.5x | Required |
| 3 | Breakout magnitude > 2% | Required |
| 4 | Close in favorable half of range | Required |
| 5 | 9 EMA aligned with breakout direction | Required |
| 6 | SPY not in strong opposite trend | Required |

### Intensity Score (0-100) for Sizing
Additional metrics that predict follow-through magnitude:

| Metric | High Score Threshold | Weight |
|--------|---------------------|--------|
| Consolidation tightness (range/ATR) | < 1.5x ATR | 20% |
| RVOL on breakout | > 3x | 20% |
| Prior volume dry-up | RVOL < 0.7 in last 3 days | 15% |
| % above/below breakout level at close | > 4% | 15% |
| Days since last touch of breakout level | > 10 days | 15% |
| Sector relative strength | Top quartile | 15% |

**Intensity Score Interpretation**:
- 80-100: Full size
- 60-79: 75% size
- 40-59: 50% size (watch for confirmation)
- < 40: Skip or 25% size only

---

## Setup Types

### Type A: Clean Consolidation Breakout
- Tight range for 7-15 days
- Volume dries up in consolidation
- Explosive breakout with volume
- **Best for**: Trend continuation, momentum plays

### Type B: Flag/Pennant Breakout
- Sharp move up/down, then consolidation (3-7 days)
- Consolidation is tighter range after big move
- Breakout resumes prior direction
- **Best for**: Momentum continuation

### Type C: Range Breakout (Multi-week)
- Consolidation > 20 days
- Well-defined support/resistance (3+ touches)
- Breakout typically more significant but slower
- **Best for**: Swing trades, larger targets

### Type D: News/Catalyst Breakout
- Breakout driven by news (earnings, upgrade, etc.)
- May have gap at open
- Volume spike on news
- **Best for**: Day trades, quick momentum

---

## Entry Rules

### Primary Entry: Breakout Confirmation
- Entry: When price breaks above resistance (long) or below support (short)
- Confirmation: 5-minute close beyond level OR 1-minute sustained break with volume
- Risk: False breakout — use invalidation level

### Secondary Entry: Pullback to Breakout Level
- Entry: Pullback to prior resistance-turned-support (or vice versa)
- Requirements: 
  - Must hold the level on first test
  - Volume should be lower than breakout
  - Time window: Same day or next 1-2 days

---

## Exit Rules

### Target Framework (ATR-based)

| Target | Calculation | Notes |
|--------|-------------|-------|
| Target 1 (Quick) | Entry + (0.5 × ATR) | Take 25-50% position |
| Target 2 (Base) | Entry + (1.0 × ATR) | Take 25-50% position |
| Target 3 (Extended) | Entry + (2.0 × ATR) | Runner only |
| Target 4 (Home run) | 2x the consolidation range | Full move expectation |

### Stop Loss
- Hard stop: Breakout level - 0.5 ATR (false break invalidation)
- Trailing stop: Below 9 EMA once Target 1 hit
- Time stop: If no follow-through in 30 min, reduce size

---

## False Breakout Warning Signs

Watch for these — consider reducing size or skipping:

1. **No volume expansion** (RVOL < 1.3x)
2. **Immediate rejection** (price back inside range within 15 min)
3. **Weak close** (close near breakout level or worse)
4. **Opposite market trend** (SPY strongly against position)
5. **Into major resistance** (prior highs, 200 MA, psych level)
6. **Low float + social pump** (meme stock dynamics)

---

## Data Collection Requirements

For each breakout setup, record:

```
- Date, ticker, setup type (A/B/C/D)
- Consolidation days, range metrics
- Breakout day: open, high, low, close, volume, RVOL
- Entry price, stop price, target prices
- Follow-through: max favorable excursion, close % from entry
- Outcome: Win/Loss, R-multiple, time in trade
- Notes: catalyst, market conditions, false break signals
```

---

## Next Steps

1. **Historical scanner**: Use these criteria to scan 2024-2025 data
2. **Manual review**: Validate 50+ examples, adjust thresholds
3. **Build scorer**: Implement GO/CAUTION/NO-GO logic
4. **Backtest**: Test entry/exit rules on collected dataset

---

*Document version: 1.0*
*Created: 2026-02-21*
*Author: Buck (Daily Task Engine)*
