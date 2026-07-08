# IV-Timing Study — Implementation Plan

**Research question:** Does implied-volatility acceleration/movement predict *when* a parabolic stock tops out?

**Nature:** Descriptive event study / hypothesis generator. n is too small for ML — the deliverable is aligned-event-time trajectories, one key distributional statistic, and effect-size tables. No classifiers, no train/test.

**Lives in:** `backtester/iv_study/` (mirrors the `options_replay/` multi-file pattern), with a pickle-cached fetch layer reusing `options_replay/theta_client.py`. Data → `data/iv_study/`, report → `reports/iv_study/`.

**Feasibility basis (verified 2026-07-07):**
- Theta Terminal v3 REST on `http://localhost:25503` (NOT v2/25510). STANDARD subscription covers IV/greeks back to ~2016 → 120/121 trades covered.
- Endpoint: `GET /v3/option/history/greeks/first_order` (1m/5m/30m/1h) returns per-bar `implied_vol, iv_error, greeks, bid, ask, underlying_price` with synchronized timestamps.
- Live-probed: MRNA 2020-02-27 full 391-bar 1m session (IV ~178%); AMD 2018-09-05 at 5m.
- No bulk chain-IV endpoint — discover ATM strikes via chain snapshot (list/strikes & list/expirations are ALL-TIME, not as-of-date).
- Dataset: 121 trades, 64% topped PREMARKET (options don't trade → IV can't time those); 44 RTH tops (~40 optionable); only 14 topped after 10:00 with real lead room. 10 microcaps unusable.

---

## 0. The core measurement problem, resolved up front

Every design choice below flows from one fact: **a parabolic underlying traverses 20–40% intraday.** That breaks the naive "track one option contract's IV" approach — a strike that is ATM at 9:30 slides deep ITM/OTM by the top, and its IV then reflects the *skew surface at that moneyness*, not the at-the-money vol we care about. Skew drift would masquerade as the velocity/acceleration signal we are hunting.

**Resolution — constant-moneyness ATM IV.** For each trade, pull a *ladder* of strikes spanning the day's price range (nearest expiry ≥ 5 DTE, calls and puts). At each 1-minute bar, the greeks payload already carries `underlying_price`; pick the strike nearest that bar's spot (linear-interpolating IV between the two bracketing strikes) and average put+call. That yields an ATM IV series at *constant moneyness* — the clean input for velocity/acceleration. Incremental cost is ~5–9 single-contract greeks calls instead of 1 (single-contract greeks are fast; the whole-chain snapshot is the slow call, used only once for strike discovery).

We also retain the **single fixed-9:30-ATM-strike** series as a cheap cross-check — if the two disagree wildly, moneyness drift is confirmed and we trust the ladder.

---

## 1. Two-track framing

The 64%-premarket-top problem means a single event study can't answer the question. Split it:

### Track A — Intraday timing (the direct answer)
- **Population:** RTH tops only. Optionable subset ≈ 40 trades. **Deep-dive n = 14** (topped after 10:00, real lead room). Secondary bucket of **30 tops in 9:30–10:00**.
- **t = 0:** `time_of_high_price` (100% populated, authoritative; parse *with* its embedded TZ offset — do **not** use the dirty `time_of_high_bucket` column).
- **Window:** τ = t − t_top ∈ **[−60, +30] min**, 1-minute grid.
- **Question:** does ATM IV peak, plateau, or roll its velocity/acceleration *before* price tops?
- **Honest handling of the 30 first-half-hour tops:** they have < 30 min of RTH lead, sometimes < 10. For these the only usable predictors are **prior-day-close ATM IV** and **opening-rotation IV (9:31–9:40 slope)**. Reported as a separate, underpowered bucket — never pooled with the post-10 group.

### Track B — Day-level signal (uses the premarket toppers too)
- **Population:** all optionable trades incl. premarket toppers ≈ **99**.
- **Features:** prior-day-close → open IV behavior — does IV *crush* or *spike* at the open on reversal days? IV percentile vs the ticker's own trailing days?
- **No-control-day limitation** is real (every CSV row is a reversal that happened; no non-reversal days exist by construction). **Cheap remedy — within-ticker pseudo-controls:** pull the *same ticker's* daily ATM IV on the **3–5 trading days immediately before** the trade date (reuse `premium-seller`'s `iv_history._atm_iv_for_date` pattern at a 1h snap). Question becomes: *does the reversal day's open-IV behavior look different from the same name's run-up days?* A matched within-name comparison, immune to the 400%-biotech-vs-60%-NVDA level problem.

---

## 2. IV measurement spec (resolved decisions)

| Decision | Choice | Justification |
|---|---|---|
| **Expiration** | Nearest listed expiry with **DTE ≥ 5**, target ~7 | Avoids 0–2 DTE gamma/pin and expiry-day IV artifacts while staying event-sensitive. |
| **Contracts** | ATM **call + put**, averaged, across a **5-strike ladder** bracketing the intraday range | Put/call average cancels parity/rate noise; ladder enables constant-moneyness reconstruction. |
| **Skew probe** | Also record 25-delta-put IV − ATM IV at each node | Rising put skew is a candidate leading tell; cheap given the ladder is already pulled. |
| **Re-ATM vs hold strike** | **Constant-moneyness (re-ATM each bar via ladder)** primary; fixed-9:30-strike is cross-check | A held strike drifts into skew territory as the stock runs 20–40%, contaminating velocity. |
| **Interval** | **1m** for the event window; **5m** resample as robustness | 1m catches lead; 5m confirms it isn't a 1-bar artifact. |
| **Smoothing** | 5-bar **rolling median** on raw IV, *then* first/second difference | Median kills 1m IV spikes without lagging; diff-after-smooth avoids amplifying noise. |
| **Normalization** | **Z-score each trade's IV vs its own pre-event baseline** (τ∈[−60,−30], or 9:31–9:45 for early tops); velocity in baseline-σ/min | Makes a 400% biotech and 60% NVDA comparable in "how many of its own sigmas did IV move." Percent-of-baseline secondary. |
| **Data-quality filters** | Drop bars with `iv_error ≥ 99`, `implied_vol ≤ 0`, `bid=ask=0`; drop **09:30 opening bar**; require ≥ 70% valid in-window bars or flag trade | Per feasibility probe; opening-bar IV is junk. |

**Velocity / acceleration (on smoothed, z-scored IV `z`):**
- `iv_vel[τ] = z[τ] − z[τ−1]` (σ/min); `iv_accel[τ] = iv_vel[τ] − iv_vel[τ−1]`
- Per-trade summary features: `t_iv_peak` (argmax smoothed IV in-window), **`iv_lead = t_iv_peak − t_top`** (the key quantity), `t_vel_zero_cross` (last τ<0 where velocity crosses + → −), `accel_sign_pre` (mean accel over τ∈[−15,0]), `iv_base`, `iv_at_top`, `iv_runup_slope`.

---

## 3. Analysis outputs

1. **Aligned event-time trajectory plot** per top-time bucket: median normalized IV over τ∈[−60,+30] with IQR band, **per-trade spaghetti** faint underneath, vertical line at τ=0. Separate panels for post-10 (n=14) and 9:30–10:00 (n=30).
2. **THE key statistic — distribution of `iv_lead = t_iv_peak − t_top`.** Histogram + median + IQR + sign test (fraction where IV peaks *strictly before* price). Bootstrap 90% CI on the median. Negative median = IV leads price = hypothesis supported. Reported per bucket with explicit n.
3. **Velocity/acceleration timing:** distribution of `t_vel_zero_cross − t_top`, mean `accel_sign_pre` — does IV decelerate before the price top?
4. **Incremental-value check:** Spearman of IV features (`iv_lead`, `iv_runup_slope`, `accel_sign_pre`) against baseline columns (`rvol_score`, `gap_pct`, `atr_pct_move`, `pct_change_3/15`, `vol_ratio_5min_to_pm`, `percent_of_premarket_vol`). If IV features just proxy rvol/gap, say so. Partial-correlation (not a model) of `iv_lead` vs `reversal_duration`/`reversal_open_low_pct` beyond those columns.
5. **Track B tables:** open-IV crush/spike distribution on reversal days vs matched pseudo-control days (paired, within-ticker); IV-percentile-vs-trailing-days on the trade day.
6. **Reporting standards:** every number carries its **n and an effect size** (median lead in minutes, ρ, bootstrap CI). No "significant/p<0.05" language — n=14 forbids it. `iv_lead` is **pre-registered** as primary; everything else explicitly exploratory. All attempted trades reported incl. no-data drops so attrition is visible.

---

## 4. Pipeline architecture

```
backtester/
├── iv_study/
│   ├── __init__.py
│   ├── config.py            # MIN_DTE=5, TARGET_DTE=7, LADDER_N=5, WIN_PRE=60,
│   │                        #   WIN_POST=30, INTERVAL="1m", BASELINE=(-60,-30),
│   │                        #   IV_ERR_MAX=99, PSEUDO_CONTROL_DAYS=5, thin-cap droplist
│   ├── trade_loader.py      # load+filter reversal_data.csv → Trade records
│   ├── iv_fetch.py          # ATM-ladder discovery + constant-moneyness IV series
│   ├── pseudo_controls.py   # same-ticker daily ATM IV, N days prior (Track B)
│   ├── 01_fetch_iv.py       # CLI: loop trades → parquet cache + manifest.csv
│   ├── 02_build_features.py # parquets → iv_features.csv (Track A + B features)
│   └── 03_event_study.py    # alignment, key statistic, correlations, plots, HTML
├── data/iv_study/
│   ├── {TICKER}_{YYYYMMDD}.parquet     # per-trade minute IV series
│   ├── controls/{TICKER}_{YYYYMMDD}.parquet
│   ├── iv_features.csv
│   └── manifest.csv                     # per-trade fetch status
└── reports/iv_study/
    ├── *.png
    └── iv_report.html                   # single self-contained report (mgmt_matrix.html pattern)
```

**Reuse, don't rebuild:** import `options_replay.theta_client` (greeks/expirations/chain snapshot, tz-aware, retry) and its infinite pickle cache. Add `sys.path.insert` to project root as `scripts/build_mgmt_nodes.py` does. Load CSV with `encoding='utf-8-sig'`, dates `format='%m/%d/%Y'`, and parse `time_of_high_price` keeping its offset (values carry `-05:00`/`-04:00`).

**ATM-strike discovery (the gotcha).** `list/strikes` and `list/expirations` are all-time, not as-of-date. Discover the real ladder from a chain snapshot:

```python
def discover_ladder(symbol, date_iso, spot_hint, exp):
    # one slow call, cached: NBBO chain at 09:35 for the chosen expiration
    snap = theta_client.get_chain_snapshot(symbol, date_iso, "09:35:00",
                                           max_dte=..., n_expirations=1)  # or at_time strike="*"
    traded = sorted(snap.loc[snap.expiration == exp, "strike"].unique())
    lo, hi = spot_hint * 0.80, spot_hint * 1.25   # bracket a 20-40% run, centered on 9:30 open
    return [k for k in traded if lo <= k <= hi]
```

Then per strike/right pull `get_option_greeks(..., interval="1m")`, filter junk bars, and build the constant-moneyness series:

```python
def atm_iv_series(strike_frames, right_pair=("C","P")):
    # strike_frames: {(strike,right): greeks_df} each indexed by minute w/ implied_vol, underlying_price
    # per minute: spot = median underlying_price across contracts;
    #   find two strikes bracketing spot; lininterp IV at spot; avg call & put
    # → DataFrame indexed by minute: atm_iv, spot, put_skew_25d (optional)
```

**Event alignment (Track A):**

```python
def align_to_top(iv_series, t_top, pre=60, post=30):
    # iv_series index tz-aware ET minutes; t_top tz-aware
    tau = (iv_series.index - t_top).total_seconds() / 60.0
    m = (tau >= -pre) & (tau <= post)
    out = iv_series[m].copy()
    out["tau"] = tau[m].round().astype(int)
    return out.set_index("tau")   # reindex to full [-pre,post] grid, ffill small gaps
```

**Graceful skip + manifest.** `01_fetch_iv.py` wraps each trade in try/except: terminal offline → abort whole run with a clear message (it's an external Java process); per-trade failure or thin data → log `status ∈ {ok, thin, no_data, no_exp, error}` with `n_bars`, `n_dropped`, `exp`, `strikes`, `err` to `manifest.csv` and **continue**. Expect ~20% attrition. Micro caps dropped at `trade_loader` (droplist: CODX, AHPI, APT, USWS, INDO, HUSA, USEG, CEI, IMPP, MULN).

---

## 5. Phasing with checkpoints

### Phase 1 — Pilot (validate data quality + eyeball signature)

| Trade | Role |
|---|---|
| **GME 2021-03-10** | Clean post-10 RTH top → primary Track A exercise |
| **NVDA 2024-02-12** | Large-cap, deep liquid chain → data-quality ceiling |
| **SMCI 2024-02-12** | High-IV momentum name → normalization stress test |
| **BBBY** | Meme, thinner chain → attrition realism |
| **MRNA 2020-02-27** | Topped **04:01 premarket** → **Track B-only** (validates pseudo-control path) |
| **NVDA 2024-03-08** | Second clean post-10 RTH top so Track A has two cases |

Build `config.py`, `trade_loader.py`, `iv_fetch.py`, `01_fetch_iv.py`, and a throwaway overlay plotter.

**Go / No-Go (data-usability gate, deliberately NOT a signal gate):**
- **GO** if ≥ 4/5 return a coherent constant-moneyness ATM IV series with ≥ 70% valid in-window bars, the two IV constructions (ladder vs fixed-strike) agree in shape, and IV magnitudes are sane.
- The IV-vs-price overlay is inspected **for information only** — absence of a visible lead in 5 trades is *not* a stop (n=5 can't rule out a signal), but garbage data is. This ordering prevents biasing the full run toward a hoped-for result.
- **NO-GO / revisit** if Theta greeks are systematically empty for these liquid names (→ re-examine expiration/strike discovery before spending on 99).

### Phase 2 — Full fetch
Run `01_fetch_iv.py` across all ~99 optionable trades + `pseudo_controls.py` for 3–5 prior days each. Review `manifest.csv`; confirm attrition ≈ 20% and that the post-10 bucket retains ≥ ~12 usable trades (below that, Track A is reported as anecdotal).

### Phase 3 — Features, event study, writeup
`02_build_features.py` → `03_event_study.py` → `reports/iv_study/iv_report.html`. Deliver the `iv_lead` distribution, bucketed trajectory plots, incremental-value correlations, Track B paired tables, and a plain-English verdict with all n's and effect sizes.

---

## Scope, runtime, failure modes

- **Scope:** ~7 source files, **~700–1,000 LOC** (fetch/alignment is the bulk; plotting ~150).
- **Runtime:** pilot a few minutes. Full fetch ≈ 99 trades × (~5 strikes × 2 rights + 1 chain snapshot) ≈ **1,000–1,500 API calls first run → ~10–25 min**, then near-instant on the infinite pickle cache. Pseudo-controls add ~99 × 5 fast 1h calls.
- **Failure modes & mitigations:** terminal offline → hard abort with message; thin small-caps → manifest `thin`, dropped from Track A, kept in Track B if any daily mark exists; strike-ladder misses → discovery via chain snapshot, not `list/strikes`; premarket tops → Track B only, never forced into Track A; 0-DTE contamination → `min_dte=5` gate; IV scale (decimal vs %) → normalize at load, assert median IV ∈ [0.1, 8]; skew drift → solved by constant-moneyness ladder, verified against fixed-strike series; **small-n honesty** → the n=14 post-10 bucket is framed as hypothesis-generating throughout, never as a validated edge.

**Caveat flagged for approval:** the strongest, most-tradeable form of the question (post-10 RTH tops, where a trader could actually act on an IV tell) has only 14 trades. This study can *generate and shape* a hypothesis and rule out gross data problems; it cannot *confirm* an edge at that n. If the `iv_lead` median lands negative with a tight bootstrap CI, the right next step is forward collection, not deployment.
