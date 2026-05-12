"""
Trading rules reference — collapsible section.
Extracted from generate_report.py HEADER_HTML, rendered with Streamlit expanders.
"""

from __future__ import annotations

import streamlit as st


def render_trading_rules():
    """Render trading rules and reference data in collapsible expanders."""
    with st.expander("Trading Rules & Reference", expanded=False):
        _render_reversal_rules()
        st.divider()
        _render_bounce_rules()
        st.divider()
        _render_cheat_sheet()
        st.divider()
        _render_general_rules()


def _render_reversal_rules():
    st.markdown("### Reversal Setup Scoring Guide")
    st.markdown("""
Each stock is scored on **5 pre-trade criteria** (cap-adjusted thresholds):

1. **9EMA Distance** - Price elevated above 9-day EMA
2. **Range (ATR)** - Prior day range vs ATR
3. **RVOL** - Volume vs 20-day average
4. **3-Day Run-Up** - Momentum into the top
5. **Gap Up** - Gap up on reversal day
""")

    st.markdown("**Historical Performance by Score (50 Grade A Trades)**")
    st.markdown("""
| Score | Trades | Win Rate | Avg P&L | Recommendation |
|-------|--------|----------|---------|----------------|
| **5/5** | 24 | 100% | +15.5% | **GO** |
| **4/5** | 14 | 93% | +14.6% | **GO** |
| **3/5** | 8 | 88% | +14.6% | **CAUTION** |
| **<3** | 4 | 50% | +1.1% | **NO-GO** |
""")

    st.markdown("**Target Price Levels (from OPEN)**")
    st.markdown("""
| Cap | Tier 1 (33%) | Tier 2 (33%) | Tier 3 (34%) |
|-----|-------------|-------------|-------------|
| Large | Gap Fill (100%) | 1.5x ATR (86%) | 2.0x ATR (57%) |
| ETF | 1.0x ATR (100%) | 1.5x ATR (80%) | 2.0x ATR (80%) |
| Medium | Gap Fill (79%) | 1.5x ATR (86%) | 2.0x ATR (69%) |
| Small | 1.0x ATR (80%) | 1.5x ATR (80%) | 2.0x ATR (80%) |
| Micro | 1.5x ATR (100%) | 2.0x ATR (100%) | 2.5x ATR (100%) |
""")
    st.caption("Squeeze Risk: ETF +0.4% | Large +2.2% | Small/Medium +10% | Micro +14% above open before reversal")


def _render_bounce_rules():
    st.markdown("### Bounce Setup Scoring Guide")
    st.markdown("""
Stocks **not above all major moving averages** (10/20/50 and 200 if available) are evaluated as bounce candidates. Auto-classified into two profiles:

| Profile | Description | Win Rate | Avg P&L |
|---------|-------------|----------|---------|
| **GapFade_weakstock** | Stock already in downtrend | 92% | +12.9% |
| **GapFade_strongstock** | Healthy stock hit by sudden selloff | 97% | +10.9% |
""")
    st.warning("IntradayCapitch pattern = AVOID. 11% WR, -10.2% avg.")

    st.markdown("**6 Pre-Trade Criteria (V3, profile-adjusted)**")
    st.markdown("""
1. **Deep Selloff** - Total % decline from recent high
2. **Discount from 30d High** - How far off recent highs
3. **Capitulation Gap Down** - Gap down on bounce day
4. **Prior Day Range Expansion** - Prior day range >= 1.0x ATR
5. **3-Day Momentum Crash** - Short-term price collapse
6. **Discount from 52wk High** - Distance from yearly peak
""")

    st.markdown("**Bounce Target Price Levels (ABOVE Open)**")
    st.markdown("""
| Cap | Tier 1 (33%) | Tier 2 (33%) | Tier 3 (34%) | n |
|-----|-------------|-------------|-------------|---|
| ETF | 0.5x ATR (87%) | 1.0x ATR (87%) | Gap Fill (53%) | 15 |
| Medium | 0.5x ATR (91%) | 1.0x ATR (78%) | Gap Fill (73%) | 45 |
| Small | 0.5x ATR (75%) | 1.0x ATR (75%) | Gap Fill (86%) | 8 |
| Large | 0.5x ATR (100%) | 1.0x ATR (94%) | Gap Fill (69%) | 16 |
""")

    st.markdown("**Bounce Intensity Score (composite 0-100)**")
    st.markdown("""
| Intensity | N | Win Rate | Avg P&L | Avg High |
|-----------|---|----------|---------|----------|
| **80+** | 6 | 100% | +35.2% | +54.4% |
| **70-80** | 4 | 100% | +13.1% | +29.1% |
| **60-70** | 9 | 89% | +7.8% | +18.8% |
| **50-60** | 7 | 100% | +9.5% | +13.8% |
| **<50** | 28 | 54% | -1.8% | +8.5% |

Key threshold: Intensity >= 50 = 96% WR, +14.8% avg P&L. Below 50 = 70% WR, +2.3% avg.
""")


def _render_cheat_sheet():
    st.markdown("### Bounce Day Cheat Sheet")

    st.markdown("**Key Decision Rules**")
    st.markdown("""
| Rule | Data |
|------|------|
| Take profits on the way up | Only 63% of open-to-high retained at close |
| First 30-min low = CRITICAL | 99% close green when low is in first 30 min |
| Cluster days > solo | Cluster: 96% WR, +12.7% avg. Solo: 79% WR, +6.7% avg |
| Exhaustion gap = much better | With: 93% WR, +11.7%. Without: 85% WR, +8.3% |
| Weak stock setups bounce harder | Weakstock med high +21.0%. Strongstock med high +12.0% |
| Near 52-week low = bigger bounce | Near 52wk low: +13.5% avg. Not near: +9.2% avg |
""")

    st.markdown("**Overnight Hold** (cluster days: 98% gapped up next morning)")
    st.markdown("""
| Metric | Cluster Days | All Trades |
|--------|-------------|------------|
| Overnight positive % | **98%** | 89% |
| Median overnight | **+14.2%** | +11.3% |
""")


def _render_general_rules():
    st.markdown("### General Rules")
    st.markdown("""
1. Quality in everything -- end day with quality & take breaks
2. Push size in liquid names
3. It's ok to consciously risk 30-40K on bread and butter / ETF aggression
4. Let the upside take care of itself
5. Selectivity -- trust your instincts -- reactive trades always best
6. Use the 2-minute bar for high volume good news / 1 min for scalp -- after VOLUME
7. Liquidity focus
8. Who gets paid? That's my trade
9. Expected Value over First Prints -- Push size in your bread and butter
""")

    st.markdown("### Morning Checklist")
    st.markdown("""
- Read overnight news
- Look at all stocks gapping up or down 5%+ (Stockfetcher, MAT, NLRTs)
- Go through rules and reminders
- Check Trump schedule
- Create one explicit process-oriented goal for the day
- Go through all events / ECO for the day
- Write down any tasks you want to accomplish today
""")
