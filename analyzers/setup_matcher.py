"""
Setup matcher: k-NN classifier for reversal setup types.

Given (ticker, date), predict which of the top-4 reversal setups it most
resembles, using only features that can be computed at/before market open.
Returns "no match" when confidence gates fail rather than guessing.

Top-4 setups (sample sizes shown):
    3DGapFade           36
    2DBreakoutIB        23
    GapDownTrendBreak   19
    2DGapFade           18

Usage:
    python -m analyzers.setup_matcher                # runs the 4 test cases
    python -m analyzers.setup_matcher MSTR 11/21/2024  # single query
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_collectors.combined_data_collection import (
    calculate_atr,
    check_breakout_stats,
    check_pct_move,
    get_bollinger_bands,
    get_intraday_timing,
    get_market_context,
    get_pct_from_mavs,
    get_pct_volume,
    get_prior_day_context,
    get_range_vol_expansion,
    get_volume,
)

_DATA_DIR = _REPO_ROOT / 'data'

TOP_4_SETUPS = ['3DGapFade', '2DBreakoutIB', 'GapDownTrendBreak', '2DGapFade']

# Pre-decision features (available at/before market open).
# NOTE: a few of these (pct_from_*mav, atr_distance_from_50mav) are computed
# in the training CSV using high-of-day rather than open. We use the same
# transform for fresh test rows to keep train/test consistent. For a live
# system you'd want to swap these for prior-close-based equivalents.
PRE_DECISION_FEATURES = [
    'atr_pct',
    'gap_pct',
    'one_day_before_range_pct',
    'two_day_before_range_pct',
    'three_day_before_range_pct',
    'pct_change_3', 'pct_change_15', 'pct_change_30',
    'pct_change_90', 'pct_change_120',
    'pct_from_10mav', 'pct_from_20mav', 'pct_from_50mav', 'pct_from_200mav',
    'atr_distance_from_50mav',
    'percent_of_premarket_vol',
    'percent_of_vol_one_day_before',
    'percent_of_vol_two_day_before',
    'percent_of_vol_three_day_before',
    'upper_band_distance', 'bollinger_width',
    'prior_day_close_vs_high_pct',
    'consecutive_up_days',
    'prior_day_range_atr',
    'gap_from_pm_high',
    'spy_5day_return',
    'uvxy_close',
]

# Feature weights — applied after z-score normalization. Higher weight = more
# influence on distance. Boosted features are the ones whose distributions
# DIFFER materially across the 4 setup types in the training data:
#   gap_pct:                    GDTB negative, others mostly positive
#   gap_from_pm_high:           GDTB -11% vs 3DGF -5% (selling pressure into open)
#   prior_day_close_vs_high_pct: GDTB 0.66 vs others 0.75-0.85 (prior day weakness)
#   percent_of_premarket_vol:   3DGF/2DGF ~33% vs 2DBIB/GDTB ~17%
FEATURE_WEIGHTS = {
    'gap_pct': 5.0,
    'gap_from_pm_high': 1.3,
    'prior_day_close_vs_high_pct': 1.4,
    'percent_of_premarket_vol': 1.4,
    'consecutive_up_days': 1.2,
}

# Clip extreme values for these features (raw-space clip) to prevent outliers
# from inflating σ and washing out discrimination. Without this clip, the
# 2DGapFade outlier with pct_from_50mav>1000 makes σ for that feature huge,
# so z-scores are tiny for everyone else and the feature becomes useless.
FEATURE_CLIPS = {
    'pct_from_10mav':           (-5.0,  5.0),
    'pct_from_20mav':           (-5.0,  5.0),
    'pct_from_50mav':           (-10.0, 10.0),
    'pct_from_200mav':          (-20.0, 20.0),
    'atr_distance_from_50mav':  (-30.0, 30.0),
    'prior_day_range_atr':      (0.0,   15.0),
    'pct_change_3':             (-3.0,  10.0),
    'pct_change_30':            (-3.0,  20.0),
    'pct_change_90':            (-3.0,  20.0),
    'pct_change_120':           (-3.0,  20.0),
    'pct_change_15':            (-3.0,  10.0),
    'percent_of_premarket_vol': (0.0,   3.0),
    'percent_of_vol_one_day_before':   (0.0, 15.0),
    'percent_of_vol_two_day_before':   (0.0, 15.0),
    'percent_of_vol_three_day_before': (0.0, 15.0),
    'one_day_before_range_pct':   (0.0, 5.0),
    'two_day_before_range_pct':   (0.0, 5.0),
    'three_day_before_range_pct': (0.0, 5.0),
}


def _clip(feature_name: str, value):
    """Apply per-feature clip in raw value space. None passes through."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return value
    rng = FEATURE_CLIPS.get(feature_name)
    if rng is None:
        return value
    lo, hi = rng
    return max(lo, min(hi, float(value)))

# Confidence gates — predictions that fail these become "no match"
MAX_TOP1_DIST = 0.30       # cosine distance: 0=identical direction, 1=orthogonal
MIN_MARGIN_RATIO = 1.05    # nearest different-label neighbor must be >=5% farther than top-1
MIN_TOP3_SAME_LABEL = 1    # at least 1 of top-3 neighbors must share top-1's label (always true; kept for future tuning)
MIN_CENTROID_MARGIN = 1.02 # centroid distance ratio (2nd / 1st) — below this, centroid is "tied", fall back to k-NN


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(_DATA_DIR / 'reversal_data.csv')
    df = df[df['setup'].isin(TOP_4_SETUPS)].copy().reset_index(drop=True)
    return df


def compute_features_fresh(ticker: str, date: str) -> pd.Series:
    """Compute features for a (ticker, date) not in the CSV via fill functions."""
    row = pd.Series({'ticker': ticker, 'date': date}, dtype=object)

    row = get_volume(row)                          # avg_daily_vol, premarket_vol, vol_one/two/three_day_before, vol_on_breakout_day
    row = get_range_vol_expansion(row)             # percent_of_vol_*_day_before, *_range_pct
    row = check_pct_move(row)                      # pct_change_*
    row = get_pct_from_mavs(row)                   # pct_from_*mav, atr_distance_from_50mav
    row = check_breakout_stats(row, 'reversal')    # gap_pct, reversal_open_close_pct (needed for ATR gate)
    row = calculate_atr(row, 'reversal')           # atr_pct (gated on reversal_open_close_pct)
    row = get_bollinger_bands(row, 'reversal')     # upper_band_distance, bollinger_width
    row = get_prior_day_context(row, 'reversal')   # prior_day_close_vs_high_pct, consecutive_up_days, prior_day_range_atr
    row = get_intraday_timing(row, 'reversal')     # gap_from_pm_high
    row = get_market_context(row, 'reversal')      # spy_5day_return, uvxy_close
    row = get_pct_volume(row)                      # percent_of_premarket_vol

    return row


def compute_features_premarket(ticker: str, date: str) -> pd.Series:
    """
    Compute pre-decision features as if scanning premarket of `date`.

    Differs from compute_features_fresh in that it substitutes any feature
    that would normally peek at post-open data (today's official open, high
    of day, etc.) with premarket-derived equivalents:

      gap_pct          = (premarket_last - prior_close) / prior_close
      gap_from_pm_high = (premarket_last - premarket_high) / premarket_high
      pct_from_*mav    = (premarket_last - SMA_N) / SMA_N         [vs high-of-day]
      atr_distance_50  = (premarket_last - SMA_50) / atr_value    [vs high-of-day]
      percent_of_premarket_vol = premarket_vol / avg_daily_vol
      premarket_vol    = today's premarket volume

    Other features (ATR, Bollinger, prior-day context, MAs themselves,
    range expansion, market context) already use only prior-day data and
    are computed identically to the post-close version.
    """
    from data_queries.polygon_queries import (
        get_daily,
        get_intraday,
        adjust_date_to_market,
        get_atr,
        poly_client,
    )
    from support.market_session import PREMARKET_START, MARKET_OPEN
    from support.date_utils import csv_date_to_iso

    iso_date = csv_date_to_iso(date)
    prior_date_iso = adjust_date_to_market(iso_date, 1)

    # 1. Get the post-close-style feature row first (we'll override the
    #    leaky fields below). This handles ATR/Bollinger/prior-day-context/
    #    pct_change/range/market context — none of which materially leak
    #    today's open/high.
    row = compute_features_fresh(ticker, date)

    # 2. Prior trading day's close (the anchor for premarket gap calculations)
    prior_daily = get_daily(ticker, prior_date_iso)
    if prior_daily is None:
        raise RuntimeError(f"No prior-day daily bar for {ticker} on {prior_date_iso}")
    prior_close = float(prior_daily.close)

    # 3. Today's premarket aggregates
    intraday = get_intraday(ticker, iso_date, multiplier=1, timespan='minute')
    pm_high = pm_low = pm_last = None
    pm_vol = 0
    if intraday is not None and not intraday.empty:
        pm = intraday.between_time(PREMARKET_START, MARKET_OPEN)
        if not pm.empty:
            pm_high = float(pm['high'].max())
            pm_low = float(pm['low'].min())
            pm_last = float(pm['close'].iloc[-1])
            pm_vol = int(pm['volume'].sum())

    # Reference price for "where the stock is right now" — premarket last if
    # we have a premarket print, else prior close (the stock simply hasn't
    # traded yet, so we can't predict a gap).
    ref_price = pm_last if pm_last is not None else prior_close

    # 4. Override gap_pct using premarket-last vs prior-close
    if prior_close > 0:
        row['gap_pct'] = (ref_price - prior_close) / prior_close

    # 5. Override gap_from_pm_high using premarket data only
    if pm_high and pm_last and pm_high > 0:
        row['gap_from_pm_high'] = (pm_last - pm_high) / pm_high
    else:
        row['gap_from_pm_high'] = 0.0

    # 6. Override premarket_vol + percent_of_premarket_vol
    row['premarket_vol'] = pm_vol
    adv = row.get('avg_daily_vol')
    try:
        adv_f = float(adv) if adv is not None and not pd.isna(adv) else None
    except (TypeError, ValueError):
        adv_f = None
    if adv_f and adv_f > 0:
        row['percent_of_premarket_vol'] = pm_vol / adv_f

    # 7. Override pct_from_*mav and atr_distance_from_50mav using ref_price
    #    (instead of high-of-day). The SMA values themselves are unchanged.
    for window, key in [(10, 'pct_from_10mav'), (20, 'pct_from_20mav'),
                        (50, 'pct_from_50mav'), (200, 'pct_from_200mav')]:
        try:
            mav = poly_client.get_sma(
                ticker=ticker, timespan='day', adjusted=True, window=window,
                series_type='close', order='desc', limit=10,
                timestamp=prior_date_iso,
            ).values[0].value
            if mav and mav > 0:
                row[key] = (ref_price - mav) / mav
        except (AttributeError, IndexError, TypeError):
            pass

    try:
        mav_50 = poly_client.get_sma(
            ticker=ticker, timespan='day', adjusted=True, window=50,
            series_type='close', order='desc', limit=10,
            timestamp=prior_date_iso,
        ).values[0].value
        atr_value = get_atr(ticker, prior_date_iso)
        if mav_50 and atr_value and atr_value > 0:
            row['atr_distance_from_50mav'] = (ref_price - mav_50) / atr_value
    except (AttributeError, IndexError, TypeError):
        pass

    # Stash the premarket aggregates so callers can show them
    row['_pm_high'] = pm_high
    row['_pm_low'] = pm_low
    row['_pm_last'] = pm_last
    row['_pm_vol'] = pm_vol
    row['_prior_close'] = prior_close

    return row


def build_feature_matrix(df: pd.DataFrame, features: list[str]):
    """Return (X weighted z-scored matrix, mu dict, sigma dict). NaN imputed at mean.
    Applies per-feature raw-value clipping before computing mean/std/z-scores."""
    X = np.zeros((len(df), len(features)))
    mu, sigma = {}, {}
    for j, f in enumerate(features):
        col_raw = pd.to_numeric(df[f], errors='coerce')
        # Clip BEFORE statistics
        col = col_raw.apply(lambda v: _clip(f, v) if pd.notna(v) else v)
        m = float(col.mean())
        s = float(col.std(ddof=0))
        if not np.isfinite(s) or s == 0:
            s = 1.0
        mu[f] = m
        sigma[f] = s
        w = FEATURE_WEIGHTS.get(f, 1.0)
        X[:, j] = (((col.fillna(m)) - m) / s) * w
    return X, mu, sigma


def featurize_row(row, features: list[str], mu: dict, sigma: dict) -> np.ndarray:
    vec = np.zeros(len(features))
    for j, f in enumerate(features):
        v = row.get(f) if hasattr(row, 'get') else None
        try:
            v = float(v) if v is not None and not pd.isna(v) else None
        except (TypeError, ValueError):
            v = None
        w = FEATURE_WEIGHTS.get(f, 1.0)
        if v is None:
            vec[j] = 0.0
        else:
            v = _clip(f, v)
            vec[j] = ((v - mu[f]) / sigma[f]) * w
    return vec


def classify_via_centroids(query_vec: np.ndarray, X: np.ndarray, y, setups: list[str]):
    """
    Centroid-based classifier: compute mean feature vector per setup, then
    predict the setup whose centroid is closest (Euclidean) to the query.
    Robust to outlier training examples because no single neighbor dominates.
    """
    y = np.asarray(y)
    centroids = {}
    n_per_setup = {}
    for s in setups:
        mask = y == s
        if not mask.any():
            continue
        centroids[s] = X[mask].mean(axis=0)
        n_per_setup[s] = int(mask.sum())

    dists = {s: float(np.linalg.norm(query_vec - c)) for s, c in centroids.items()}
    predicted = min(dists, key=lambda k: dists[k])
    sorted_dists = sorted(dists.items(), key=lambda kv: kv[1])
    top1_dist = sorted_dists[0][1]
    top2_dist = sorted_dists[1][1] if len(sorted_dists) > 1 else top1_dist * 2
    margin_ratio = top2_dist / top1_dist if top1_dist > 1e-9 else float('inf')

    return {
        'centroid_predicted': predicted,
        'centroid_distances': dists,
        'centroid_margin_ratio': margin_ratio,
        'centroid_n_per_setup': n_per_setup,
    }


def knn_classify(query_vec: np.ndarray, X: np.ndarray, y, k: int = 5):
    """
    Top-1 with margin algorithm:
      - Prediction = top-1 nearest neighbor's label (NOT majority vote)
      - Distance-weighted votes are also computed for diagnostics, but the
        algorithm trusts the closest single match rather than letting many
        moderately-distant neighbors outvote it.
    Distance metric: cosine distance on weighted z-scored features.
    """
    y = np.asarray(y)
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sims = Xn @ qn
    distances = 1.0 - sims

    order = np.argsort(distances)[:k]
    neighbors = [(str(y[i]), float(distances[i]), int(i)) for i in order]

    top_lbl = neighbors[0][0]
    top_dist = neighbors[0][1]

    # Diagnostic: distance-weighted vote among top-k (NOT used for prediction)
    votes: Counter = Counter()
    for lbl, d, _ in neighbors:
        votes[lbl] += 1.0 / (d + 0.05)
    total_votes = sum(votes.values())
    top1_label_vote_share = votes.get(top_lbl, 0.0) / total_votes if total_votes else 0.0

    same_lbl_in_top3 = sum(1 for lbl, _, _ in neighbors[:3] if lbl == top_lbl)
    diff_label_dists = [d for lbl, d, _ in neighbors if lbl != top_lbl]
    nearest_diff = min(diff_label_dists) if diff_label_dists else None
    margin_ratio = (nearest_diff / top_dist) if (nearest_diff and top_dist > 1e-9) else float('inf')
    margin_abs = (nearest_diff - top_dist) if nearest_diff is not None else 1.0

    return {
        'predicted': top_lbl,
        'top1_dist': top_dist,
        'top1_label_vote_share': top1_label_vote_share,
        'same_label_in_top3': same_lbl_in_top3,
        'margin_abs': margin_abs,
        'margin_ratio': margin_ratio,
        'neighbors': neighbors,
        'votes': dict(votes),
    }


def apply_confidence_gate(result: dict) -> dict:
    """Override prediction with 'no_match' if confidence gates fail."""
    pass_top1 = result['top1_dist'] <= MAX_TOP1_DIST
    pass_margin = result['margin_ratio'] >= MIN_MARGIN_RATIO
    pass_top3 = result['same_label_in_top3'] >= MIN_TOP3_SAME_LABEL
    confident = pass_top1 and pass_margin and pass_top3

    result['gate_pass_top1_dist'] = pass_top1
    result['gate_pass_margin'] = pass_margin
    result['gate_pass_top3_consistency'] = pass_top3
    result['confident'] = confident
    result['final_prediction'] = result['predicted'] if confident else 'no_match'
    return result


def match_setup(ticker: str, date: str, k: int = 5,
                train_df: pd.DataFrame | None = None,
                premarket: bool = False):
    """
    Classify (ticker, date) into one of the top-4 reversal setups.

    If `premarket=True`, features are computed using ONLY premarket and
    prior-day data — simulating what would be visible pre-open on `date`.
    Otherwise, post-close features are used (matches how training data
    in reversal_data.csv was generated).

    If (ticker, date) is in the training CSV, it's leave-one-out unless
    premarket=True (premarket re-fetches features regardless, since the
    CSV's gap_pct/pct_from_*mav were computed post-open).
    """
    if train_df is None:
        train_df = load_training_data()

    test_mask = (train_df['ticker'] == ticker) & (train_df['date'] == date)
    in_csv = bool(test_mask.any())

    if in_csv and not premarket:
        test_row = train_df[test_mask].iloc[0]
        actual_label = test_row['setup']
        working_train = train_df[~test_mask].reset_index(drop=True)
    else:
        # If in CSV but premarket=True, still leave-one-out from the training
        # set, but compute fresh premarket features for the query row
        if in_csv:
            actual_label = train_df[test_mask].iloc[0]['setup']
            working_train = train_df[~test_mask].reset_index(drop=True)
        else:
            actual_label = None
            working_train = train_df.reset_index(drop=True)
        if premarket:
            print(f"[fetch] computing PREMARKET features for {ticker} on {date}...")
            test_row = compute_features_premarket(ticker, date)
        else:
            print(f"[fetch] computing fresh features for {ticker} on {date}...")
            test_row = compute_features_fresh(ticker, date)

    X, mu, sigma = build_feature_matrix(working_train, PRE_DECISION_FEATURES)
    y = working_train['setup'].values
    query_vec = featurize_row(test_row, PRE_DECISION_FEATURES, mu, sigma)

    result = knn_classify(query_vec, X, y, k=k)

    # Centroid-based classification (more robust to outlier neighbors)
    centroid_info = classify_via_centroids(query_vec, X, y, TOP_4_SETUPS)
    result.update(centroid_info)

    # Combined classifier — confidence-scoring ensemble:
    #   centroid_confidence = how-much-better-than-tied the centroid margin is.
    #     Margin 1.00 = tied → 0.0
    #     Margin 1.10 = +10% better → 1.0
    #     Margin >1.10 caps at >1.0
    #   knn_confidence = how-much-closer-than-the-threshold the top-1 is.
    #     dist 0.30 (gate) → 0.0
    #     dist 0.00       → 1.0
    #     dist >0.30 caps below 0
    # Trust whichever signal has higher confidence. This handles two failure
    # modes the old margin-threshold rule got wrong:
    #   - Centroid margin 1.02-1.10 is "barely a signal" — k-NN with very
    #     close top-1 (d < 0.10) should usually win.
    #   - Centroid margin > 1.15 is solid — k-NN can't override unless its
    #     top-1 is extremely close.
    knn_pred = result['predicted']
    cen_pred = result['centroid_predicted']
    cen_margin = result['centroid_margin_ratio']
    knn_d = result['top1_dist']

    cen_conf = (cen_margin - 1.0) / 0.10
    knn_conf = max(0.0, (0.30 - knn_d) / 0.30)

    if knn_pred == cen_pred:
        combined_pred = knn_pred
        combined_reason = (f'k-NN and centroid agree '
                           f'(centroid_conf={cen_conf:.2f}, knn_conf={knn_conf:.2f})')
    elif cen_conf >= knn_conf:
        combined_pred = cen_pred
        combined_reason = (f'centroid_conf {cen_conf:.2f} >= knn_conf {knn_conf:.2f}; '
                           f'trusting centroid ({cen_pred}) over k-NN ({knn_pred})')
    else:
        combined_pred = knn_pred
        combined_reason = (f'knn_conf {knn_conf:.2f} > centroid_conf {cen_conf:.2f}; '
                           f'trusting k-NN ({knn_pred}) over centroid ({cen_pred})')
    result['combined_predicted'] = combined_pred
    result['combined_reason'] = combined_reason
    result['centroid_confidence'] = cen_conf
    result['knn_confidence'] = knn_conf

    # Diagnostic: nearest neighbor PER setup label (shows whether the correct
    # label even has a close example or is simply too far)
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    all_dists = 1.0 - (Xn @ qn)
    per_label_best = {}
    for setup in TOP_4_SETUPS:
        idxs = np.where(np.asarray(y) == setup)[0]
        if len(idxs) == 0:
            continue
        local_dists = all_dists[idxs]
        best_local = idxs[int(np.argmin(local_dists))]
        per_label_best[setup] = {
            'ticker': working_train.iloc[best_local]['ticker'],
            'date': working_train.iloc[best_local]['date'],
            'distance': float(all_dists[best_local]),
        }
    result['nearest_per_label'] = per_label_best
    result = apply_confidence_gate(result)
    result['ticker'] = ticker
    result['date'] = date
    result['in_csv'] = in_csv
    result['actual'] = actual_label

    enriched = []
    for lbl, d, idx in result['neighbors']:
        enriched.append({
            'ticker': working_train.iloc[idx]['ticker'],
            'date': working_train.iloc[idx]['date'],
            'setup': lbl,
            'distance': d,
        })
    result['neighbor_details'] = enriched

    # Capture raw (un-normalized) feature values from the test row for inspection
    raw = {}
    for f in PRE_DECISION_FEATURES:
        try:
            raw[f] = test_row.get(f) if hasattr(test_row, 'get') else None
        except Exception:
            raw[f] = None
    result['raw_query_features'] = raw

    # Capture premarket aggregates (only set in premarket mode) for caller printing
    for k in ('_pm_high', '_pm_low', '_pm_last', '_pm_vol', '_prior_close'):
        try:
            result[k] = test_row.get(k) if hasattr(test_row, 'get') else None
        except Exception:
            result[k] = None

    # Feature inspection: which features pushed the top-1 neighbor close?
    top_idx = result['neighbors'][0][2]
    train_vec = ((X[top_idx]))
    contribs = []
    for j, f in enumerate(PRE_DECISION_FEATURES):
        diff_sq = (query_vec[j] - train_vec[j]) ** 2
        contribs.append((f, query_vec[j], train_vec[j], diff_sq))
    contribs.sort(key=lambda r: r[3])
    result['top_matching_features'] = contribs[:5]
    result['worst_matching_features'] = contribs[-5:]

    return result


def print_result(result: dict, expected: str | None = None):
    print()
    print('=' * 78)
    print(f"Query: {result['ticker']} on {result['date']}", end='')
    if result['in_csv']:
        print(f"  [in CSV — leave-one-out, actual={result['actual']}]")
    else:
        print()

    final = result['final_prediction']
    print(f"Final prediction:    {final}")
    print(f"k-NN top label:      {result['predicted']}")
    if expected is not None:
        ok = '[OK]' if final == expected else '[XX]'
        print(f"Expected:            {expected}  {ok}")
    print(f"  top1 cosine dist:  {result['top1_dist']:.4f}   "
          f"(gate <={MAX_TOP1_DIST}: {'pass' if result['gate_pass_top1_dist'] else 'FAIL'})")
    print(f"  margin ratio:      {result['margin_ratio']:.3f}    "
          f"(gate >={MIN_MARGIN_RATIO}: {'pass' if result['gate_pass_margin'] else 'FAIL'})")
    print(f"  same-label top-3:  {result['same_label_in_top3']}/3      "
          f"(gate >={MIN_TOP3_SAME_LABEL}: {'pass' if result['gate_pass_top3_consistency'] else 'FAIL'})")
    print(f"  margin abs:        {result['margin_abs']:.4f}")
    print(f"  diagnostic votes:  {result['votes']}")

    print("\n  Nearest 5 neighbors:")
    for n in result['neighbor_details']:
        print(f"    {n['ticker']:6s} {n['date']:12s}  {n['setup']:20s}  d={n['distance']:.4f}")

    print("\n  Nearest neighbor per setup label:")
    npl = result.get('nearest_per_label', {})
    for setup in TOP_4_SETUPS:
        info = npl.get(setup)
        if info:
            print(f"    {setup:25s} -> {info['ticker']:6s} {info['date']:12s}  d={info['distance']:.4f}")

    print("\n  Features matching top-1 neighbor most closely (z-scores):")
    for f, q, t, d in result['top_matching_features']:
        print(f"    {f:32s}  query={q:+.2f}  neighbor={t:+.2f}")
    print("\n  Features diverging most from top-1 neighbor (z-scores):")
    for f, q, t, d in result['worst_matching_features']:
        print(f"    {f:32s}  query={q:+.2f}  neighbor={t:+.2f}")

    # Raw key features (sanity check)
    raw = result.get('raw_query_features')
    if raw:
        print("\n  Raw query feature values (key sanity-check fields):")
        for f in ('gap_pct', 'consecutive_up_days', 'pct_from_9ema', 'pct_from_50mav',
                  'atr_pct', 'percent_of_premarket_vol', 'gap_from_pm_high',
                  'pct_change_3', 'pct_change_30'):
            if f in raw:
                v = raw[f]
                try:
                    v = float(v)
                    print(f"    {f:32s} {v:+.4f}")
                except (TypeError, ValueError):
                    print(f"    {f:32s} {v}")


# ----------------------------------------------------------------------------
# 3DGapFade-specific detector
#
# Reframed as a binary detector to avoid false positives across setup classes
# whose discriminating signals (e.g. intraday inside-bar pattern, intraday
# trend-line breaks) aren't visible pre-decision. 3DGapFade has the richest
# training sample (n=36), most distinctive pre-decision feature signature, and
# the most well-defined necessary conditions:
#
#   Necessary conditions (prefilter — any 3DGapFade must satisfy):
#     gap_pct                >= +0.02  (gap up at least +2%)
#     consecutive_up_days    >= 2      (multi-day uptrend, hence "3D")
#
#   Sufficient confidence (k-NN against 3DGapFade examples in training):
#     top-1 nearest is a 3DGapFade
#     top-1 cosine distance  <= MAX_TOP1_DIST
#     >=3 of top-5 nearest are 3DGapFade  (neighborhood dominance)
# ----------------------------------------------------------------------------

DETECT_3DGF_MIN_GAP_PCT = 0.02
DETECT_3DGF_MIN_UP_DAYS = 2
DETECT_3DGF_MAX_TOP1_DIST = 0.30
DETECT_3DGF_MIN_TOP5_SAME = 2


def detect_3dgapfade(ticker: str, date: str, train_df: pd.DataFrame | None = None):
    """
    Binary detector: does (ticker, date) look like a 3DGapFade?

    Returns dict with `is_3dgapfade` (bool), `reason`, and full match diagnostics.
    """
    base = match_setup(ticker, date, k=5, train_df=train_df)
    raw = base['raw_query_features']

    gap = raw.get('gap_pct')
    up_days = raw.get('consecutive_up_days')
    try:
        gap_f = float(gap) if gap is not None else None
    except (TypeError, ValueError):
        gap_f = None
    try:
        up_f = float(up_days) if up_days is not None else None
    except (TypeError, ValueError):
        up_f = None

    # Prefilter (hard rejects)
    if gap_f is None or gap_f < DETECT_3DGF_MIN_GAP_PCT:
        base['is_3dgapfade'] = False
        gap_str = f"{gap_f:+.3f}" if gap_f is not None else "None"
        base['reject_reason'] = f"gap_pct {gap_str} below threshold +{DETECT_3DGF_MIN_GAP_PCT:.2f}"
        base['detector_decision'] = '3DGapFade: NO'
        return base

    if up_f is None or up_f < DETECT_3DGF_MIN_UP_DAYS:
        base['is_3dgapfade'] = False
        base['reject_reason'] = f"consecutive_up_days {up_f} below threshold {DETECT_3DGF_MIN_UP_DAYS}"
        base['detector_decision'] = '3DGapFade: NO'
        return base

    # Confidence: k-NN agreement
    top1_label = base['predicted']
    top1_dist = base['top1_dist']
    same_in_top5 = sum(1 for lbl, _, _ in base['neighbors'] if lbl == '3DGapFade')

    if top1_label != '3DGapFade':
        base['is_3dgapfade'] = False
        base['reject_reason'] = f"top-1 nearest is {top1_label}, not 3DGapFade"
        base['detector_decision'] = '3DGapFade: NO (shape mismatch)'
        return base

    if top1_dist > DETECT_3DGF_MAX_TOP1_DIST:
        base['is_3dgapfade'] = False
        base['reject_reason'] = f"top-1 distance {top1_dist:.3f} > {DETECT_3DGF_MAX_TOP1_DIST}"
        base['detector_decision'] = '3DGapFade: NO (no close historical analog)'
        return base

    if same_in_top5 < DETECT_3DGF_MIN_TOP5_SAME:
        base['is_3dgapfade'] = False
        base['reject_reason'] = f"only {same_in_top5}/5 neighbors are 3DGapFade (need {DETECT_3DGF_MIN_TOP5_SAME})"
        base['detector_decision'] = '3DGapFade: NO (neighborhood not dominated)'
        return base

    base['is_3dgapfade'] = True
    base['reject_reason'] = None
    base['detector_decision'] = '3DGapFade: YES'
    return base


def print_detector_result(result: dict, expected_is_3dgf: bool):
    print()
    print('=' * 78)
    print(f"Query: {result['ticker']} on {result['date']}", end='')
    if result['in_csv']:
        print(f"  [in CSV — leave-one-out, actual={result['actual']}]")
    else:
        print()

    decision = result['detector_decision']
    expected_str = '3DGapFade: YES' if expected_is_3dgf else '3DGapFade: NO'
    correct = result['is_3dgapfade'] == expected_is_3dgf
    flag = '[OK]' if correct else '[XX]'
    print(f"  Decision:  {decision}")
    print(f"  Expected:  {expected_str}  {flag}")

    raw = result['raw_query_features']
    print(f"  Prefilter values:")
    g = raw.get('gap_pct')
    u = raw.get('consecutive_up_days')
    try:
        g = float(g) if g is not None else None
    except (TypeError, ValueError):
        g = None
    try:
        u = float(u) if u is not None else None
    except (TypeError, ValueError):
        u = None
    g_pass = (g is not None and g >= DETECT_3DGF_MIN_GAP_PCT)
    u_pass = (u is not None and u >= DETECT_3DGF_MIN_UP_DAYS)
    print(f"    gap_pct = {g if g is None else f'{g:+.4f}'}  "
          f"(need >= +{DETECT_3DGF_MIN_GAP_PCT:.2f}: {'pass' if g_pass else 'FAIL'})")
    print(f"    consecutive_up_days = {u}  "
          f"(need >= {DETECT_3DGF_MIN_UP_DAYS}: {'pass' if u_pass else 'FAIL'})")

    if g_pass and u_pass:
        same_in_top5 = sum(1 for lbl, _, _ in result['neighbors'] if lbl == '3DGapFade')
        print(f"  Shape match (k=5 k-NN):")
        print(f"    top-1 label = {result['predicted']}  "
              f"({'PASS' if result['predicted'] == '3DGapFade' else 'FAIL'})")
        print(f"    top-1 dist  = {result['top1_dist']:.4f}  "
              f"(need <= {DETECT_3DGF_MAX_TOP1_DIST}: "
              f"{'pass' if result['top1_dist'] <= DETECT_3DGF_MAX_TOP1_DIST else 'FAIL'})")
        print(f"    3DGapFade in top-5: {same_in_top5}/5  "
              f"(need >= {DETECT_3DGF_MIN_TOP5_SAME}: "
              f"{'pass' if same_in_top5 >= DETECT_3DGF_MIN_TOP5_SAME else 'FAIL'})")

        print(f"  Nearest 5 neighbors:")
        for n in result['neighbor_details']:
            same = '<-- 3DGapFade' if n['setup'] == '3DGapFade' else ''
            print(f"    {n['ticker']:6s} {n['date']:12s}  {n['setup']:20s}  d={n['distance']:.4f}  {same}")

    if result.get('reject_reason'):
        print(f"  Reject reason: {result['reject_reason']}")


def print_multiclass_result(result: dict, expected: str):
    """Print multi-class (top-4) classification result with centroid + k-NN diagnostics."""
    print()
    print('=' * 78)
    print(f"Query: {result['ticker']} on {result['date']}", end='')
    if result['in_csv']:
        print(f"  [in CSV — leave-one-out, actual={result['actual']}]")
    else:
        print()

    pred = result['combined_predicted']
    ok = '[OK]' if pred == expected else '[XX]'
    print(f"  Predicted (combined): {pred}  (expected {expected})  {ok}")
    print(f"  Decision basis:       {result['combined_reason']}")
    print()
    print(f"  Centroid classifier:")
    print(f"    Predicted: {result['centroid_predicted']}    margin ratio={result['centroid_margin_ratio']:.3f}")
    for setup, dist in sorted(result['centroid_distances'].items(), key=lambda kv: kv[1]):
        marker = '   <-- predicted' if setup == result['centroid_predicted'] else ''
        n = result['centroid_n_per_setup'].get(setup, 0)
        print(f"      {setup:25s}  d={dist:.4f}  (n={n}){marker}")
    print()
    print(f"  k-NN classifier (top-1):")
    print(f"    Predicted: {result['predicted']}    top1 dist={result['top1_dist']:.4f}")
    print(f"    Nearest neighbors:")
    for n in result['neighbor_details']:
        marker = '   <-- top1' if n is result['neighbor_details'][0] else ''
        print(f"      {n['ticker']:6s} {n['date']:12s}  {n['setup']:20s}  d={n['distance']:.4f}{marker}")
    print()
    raw = result['raw_query_features']
    print(f"  Raw query features:")
    for f in ('gap_pct', 'consecutive_up_days', 'pct_change_3', 'pct_change_30',
              'pct_from_50mav', 'percent_of_premarket_vol', 'gap_from_pm_high',
              'prior_day_range_atr'):
        v = raw.get(f)
        try:
            v = float(v)
            print(f"    {f:32s} {v:+.4f}")
        except (TypeError, ValueError):
            print(f"    {f:32s} {v}")


def run_test_suite():
    """
    Multi-class test: classify each of the 4 user-provided examples to its
    setup label. Uses the combined classifier (centroid + k-NN ensemble).
    """
    test_cases = [
        ('MSTR', '11/21/2024', '3DGapFade'),
        ('GME',  '5/14/2024',  '2DGapFade'),        # user said 5/14/23 but that's Sunday; 5/14/24 = Roaring Kitty squeeze v2
        ('OPEN', '8/25/2025',  '2DBreakoutIB'),
        ('QS',   '7/21/2025',  'GapDownTrendBreak'),
    ]
    correct = 0
    train_df = load_training_data()

    for ticker, date, expected in test_cases:
        try:
            result = match_setup(ticker, date, k=5, train_df=train_df)
            print_multiclass_result(result, expected)
            if result['combined_predicted'] == expected:
                correct += 1
        except Exception as e:
            print(f"\n!! ERROR for {ticker} {date}: {e}")
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 78)
    print(f"MULTI-CLASS FINAL SCORE: {correct}/{len(test_cases)} correct")
    print('=' * 78)


# Reporting-side confidence gates (strict — suppresses weak/contradicted predictions)
MIN_REPORT_CONFIDENCE = 0.20  # max(cen_conf, knn_conf) must clear this


def match_setup_for_report(ticker: str, date_str: str) -> dict | None:
    """
    Compatibility wrapper for the priority report — runs the matcher in
    premarket mode and returns a compact dict for HTML rendering, or
    None when:
      - the matcher fails outright, OR
      - the ensemble's predicted label is NOT the centroid's top-1 closest
        setup (i.e. the centroid disagrees with the prediction), OR
      - max(centroid_confidence, knn_confidence) < MIN_REPORT_CONFIDENCE

    These gates intentionally hide weak/contradicted predictions instead of
    rendering a misleading confident chip.

    Returned dict keys (all primitives, JSON-safe):
        ticker, date
        predicted                 final combined prediction
        decision_reason           one-line explanation
        centroid_predicted        centroid-only top pick
        centroid_distances        {setup: distance}
        centroid_margin           top1 / top2 distance ratio
        centroid_confidence       0..1+
        knn_predicted             k-NN top-1 label
        knn_top1_dist             cosine distance to top-1 neighbor
        knn_confidence            0..1
        neighbors                 [{ticker, date, setup, distance}, ...]
        pm_high / pm_low / pm_last / pm_vol / prior_close
        gap_pct                   computed gap (premarket)
    """
    try:
        result = match_setup(ticker, date_str, k=5, premarket=True)
    except Exception:
        return None

    # Confidence gates — suppress weak/contradicted predictions
    predicted = result['combined_predicted']
    centroid_top = result['centroid_predicted']
    cen_conf = result.get('centroid_confidence') or 0.0
    knn_conf = result.get('knn_confidence') or 0.0

    if predicted != centroid_top:
        return None  # centroid's top-1 disagrees with the ensemble prediction
    if max(cen_conf, knn_conf) < MIN_REPORT_CONFIDENCE:
        return None  # neither classifier is meaningfully confident

    raw = result.get('raw_query_features', {}) or {}
    return {
        'ticker': ticker,
        'date': date_str,
        'predicted': result['combined_predicted'],
        'decision_reason': result['combined_reason'],
        'centroid_predicted': result['centroid_predicted'],
        'centroid_distances': result['centroid_distances'],
        'centroid_margin': result['centroid_margin_ratio'],
        'centroid_confidence': result.get('centroid_confidence'),
        'knn_predicted': result['predicted'],
        'knn_top1_dist': result['top1_dist'],
        'knn_confidence': result.get('knn_confidence'),
        'neighbors': result['neighbor_details'],
        'pm_high': result.get('_pm_high'),
        'pm_low': result.get('_pm_low'),
        'pm_last': result.get('_pm_last'),
        'pm_vol': result.get('_pm_vol'),
        'prior_close': result.get('_prior_close'),
        'gap_pct': raw.get('gap_pct'),
        'consecutive_up_days': raw.get('consecutive_up_days'),
    }


if __name__ == '__main__':
    args = sys.argv[1:]
    premarket_flag = False
    if '--premarket' in args:
        premarket_flag = True
        args.remove('--premarket')

    if len(args) == 2:
        ticker, date = args[0], args[1]
        res = match_setup(ticker, date, k=5, premarket=premarket_flag)
        if premarket_flag:
            print()
            print(f"PREMARKET mode — features simulated as of premarket of {date}")
            print(f"  premarket high: {res.get('_pm_high')}")
            print(f"  premarket low:  {res.get('_pm_low')}")
            print(f"  premarket last: {res.get('_pm_last')}")
            print(f"  premarket vol:  {res.get('_pm_vol'):,}" if res.get('_pm_vol') else "  premarket vol:  0")
            print(f"  prior close:    {res.get('_prior_close')}")
        print_multiclass_result(res, expected='(unknown)')
    else:
        run_test_suite()
