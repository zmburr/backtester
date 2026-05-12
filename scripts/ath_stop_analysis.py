"""
ATH Stop-Out Analysis for 3DGapFade setups.

Models the impact of the user's rule: stop out if stock reaches all-time high.
Uses backscanner cache to compute running ATH per ticker, then models how
capping losses at ATH would affect EV and blowup rates.
"""

import pandas as pd
import pickle
import os
import gc


def load_ticker_history(target_tickers: set, cache_dir: str) -> pd.DataFrame:
    """Load price history for target tickers from backscanner cache files."""
    cache_files = [
        os.path.join(cache_dir, 'grouped_daily_2019-02-25_2021-12-31.pkl'),
        os.path.join(cache_dir, 'grouped_daily_2021-02-25_2023-12-31.pkl'),
        os.path.join(cache_dir, 'grouped_daily_2023-02-25_2025-12-31.pkl'),
    ]

    all_rows = []
    for cf in cache_files:
        if not os.path.exists(cf):
            print(f"  SKIP {os.path.basename(cf)} (not found)")
            continue
        print(f"  Loading {os.path.basename(cf)}...")
        with open(cf, 'rb') as f:
            daily_data = pickle.load(f)

        for date_str, day_df in daily_data.items():
            if day_df is None or day_df.empty:
                continue
            mask = day_df['ticker'].isin(target_tickers)
            if mask.any():
                subset = day_df.loc[mask, ['ticker', 'open', 'high', 'close']].copy()
                subset['date'] = date_str
                all_rows.append(subset)

        del daily_data
        gc.collect()
        print(f"    Rows so far: {sum(len(r) for r in all_rows):,}")

    result = pd.concat(all_rows, ignore_index=True)
    del all_rows
    gc.collect()

    result = result.drop_duplicates(subset=['ticker', 'date'])
    result = result.sort_values(['ticker', 'date'])
    return result


def compute_ath(ticker_history: pd.DataFrame) -> pd.DataFrame:
    """Add running ATH and prior-day ATH columns."""
    ticker_history['ath'] = ticker_history.groupby('ticker')['high'].cummax()
    ticker_history['ath_prior'] = ticker_history.groupby('ticker')['ath'].shift(1)
    return ticker_history


def show_stats(sub, label, return_col='fade_day_return'):
    n = len(sub)
    if n == 0:
        print(f"  {label}: n=0")
        return
    wr = (sub[return_col] < 0).mean() * 100
    ev = -sub[return_col].mean() * 100
    med = -sub[return_col].median() * 100
    blowup = (sub[return_col] > 0.10).mean() * 100
    print(f"  {label}: n={n}, WR={wr:.0f}%, EV={ev:+.1f}%, Median={med:+.1f}%, Blowups={blowup:.0f}%")


def main():
    # Load universe
    print("Loading universe data...")
    universe = pd.read_csv('data/reversal_universe_2020-01-01_2025-12-31.csv')
    gf = universe[(universe['setup_type'] == '3DGapFade') & (universe['cap'] != 'Micro')].copy()
    target_tickers = set(gf['ticker'].unique())
    print(f"Target: {len(target_tickers)} tickers, {len(gf)} setups")

    # Load price history from cache
    print("\nLoading price history from cache...")
    cache_dir = 'data/backscanner_cache'
    history = load_ticker_history(target_tickers, cache_dir)
    print(f"Combined: {len(history):,} rows, {history['ticker'].nunique()} tickers")

    # Compute running ATH
    print("Computing running ATH...")
    history = compute_ath(history)

    # Join with setups
    history['key'] = history['ticker'] + '_' + history['date']
    gf['key'] = gf['ticker'] + '_' + gf['date']

    lookup = history.set_index('key')[['high', 'ath_prior', 'open']].rename(
        columns={'high': 'day_high', 'open': 'day_open_cache'}
    )
    gf = gf.join(lookup, on='key')
    matched = gf['ath_prior'].notna().sum()
    print(f"Matched {matched}/{len(gf)} setups with ATH data")

    gf_valid = gf[gf['ath_prior'].notna()].copy()

    # Compute ATH metrics
    gf_valid['pct_below_ath'] = (gf_valid['ath_prior'] - gf_valid['day_open_cache']) / gf_valid['ath_prior']
    gf_valid['opened_above_ath'] = gf_valid['day_open_cache'] > gf_valid['ath_prior']
    gf_valid['hit_ath_intraday'] = gf_valid['day_high'] > gf_valid['ath_prior']

    # Model ATH stop: if stock hits ATH, stop out at ATH level
    gf_valid['ath_stop_return'] = gf_valid['fade_day_return']

    # If opened below ATH but hit ATH during day: loss capped at (ATH - open) / open
    below_but_hit = (~gf_valid['opened_above_ath']) & (gf_valid['hit_ath_intraday'])
    gf_valid.loc[below_but_hit, 'ath_stop_return'] = (
        (gf_valid.loc[below_but_hit, 'ath_prior'] - gf_valid.loc[below_but_hit, 'day_open_cache'])
        / gf_valid.loc[below_but_hit, 'day_open_cache']
    )

    # --- Results ---
    above_ath = gf_valid['opened_above_ath']
    print(f"\n{'='*60}")
    print(f"ATH STOP-OUT ANALYSIS — 3DGapFade (no Micro)")
    print(f"{'='*60}")
    print(f"\nTotal valid setups: {len(gf_valid)}")
    print(f"Opened ABOVE prior ATH: {above_ath.sum()} ({above_ath.mean()*100:.0f}%)")
    print(f"Opened below, hit ATH intraday: {below_but_hit.sum()} ({below_but_hit.mean()*100:.1f}%)")
    print(f"Never reached ATH: {(~gf_valid['hit_ath_intraday'] & ~above_ath).sum()}")

    print(f"\n--- Scenario 1: Baseline (no ATH rule) ---")
    show_stats(gf_valid, "All 3DGapFade")
    show_stats(gf_valid[gf_valid['recommendation'] == 'GO'], "GO only")
    show_stats(gf_valid[gf_valid['recommendation'].isin(['GO', 'CAUTION'])], "GO+CAUTION")

    print(f"\n--- Scenario 2: SKIP trades where stock opens above ATH ---")
    below = gf_valid[~gf_valid['opened_above_ath']]
    show_stats(below, "All (below ATH)")
    show_stats(below[below['recommendation'] == 'GO'], "GO (below ATH)")
    show_stats(below[below['recommendation'].isin(['GO', 'CAUTION'])], "GO+CAUTION (below ATH)")

    skipped = gf_valid[gf_valid['opened_above_ath']]
    show_stats(skipped, "SKIPPED (above ATH)")

    print(f"\n--- Scenario 3: Stop out at ATH (keep all trades, cap loss) ---")
    show_stats(gf_valid, "All (ATH stop)", return_col='ath_stop_return')
    show_stats(gf_valid[gf_valid['recommendation'] == 'GO'], "GO (ATH stop)", return_col='ath_stop_return')
    show_stats(gf_valid[gf_valid['recommendation'].isin(['GO', 'CAUTION'])], "GO+CAUTION (ATH stop)", return_col='ath_stop_return')

    print(f"\n--- Scenario 4: Skip above-ATH + stop at ATH ---")
    combo = gf_valid[~gf_valid['opened_above_ath']].copy()
    show_stats(combo, "All (skip + stop)", return_col='ath_stop_return')
    show_stats(combo[combo['recommendation'] == 'GO'], "GO (skip + stop)", return_col='ath_stop_return')
    show_stats(combo[combo['recommendation'].isin(['GO', 'CAUTION'])], "GO+CAUTION (skip + stop)", return_col='ath_stop_return')

    # Blowup analysis
    print(f"\n--- Top 15 worst blowups: ATH stop impact ---")
    worst = gf_valid.nlargest(15, 'fade_day_return')
    print(f"  {'Ticker':<8} {'Date':<12} {'Orig':>8} {'Stopped':>8} {'ATH Status'}")
    print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8} {'-'*25}")
    for _, row in worst.iterrows():
        orig = row['fade_day_return'] * 100
        stopped_ret = row['ath_stop_return'] * 100
        if row['opened_above_ath']:
            ath_status = "ABOVE ATH (skip)"
        elif row['hit_ath_intraday']:
            ath_status = f"Hit ATH ({row['pct_below_ath']*100:.1f}% below)"
        else:
            ath_status = f"{row['pct_below_ath']*100:.0f}% below ATH"
        print(f"  {row['ticker']:<8} {row['date']:<12} {orig:>+7.1f}% {stopped_ret:>+7.1f}% {ath_status}")

    # Savings summary
    print(f"\n--- EV Impact Summary ---")
    baseline_ev = -gf_valid['fade_day_return'].mean() * 100
    skip_ev = -below['fade_day_return'].mean() * 100
    stop_ev = -gf_valid['ath_stop_return'].mean() * 100
    combo_ev = -combo['ath_stop_return'].mean() * 100
    print(f"  Baseline:                EV = {baseline_ev:+.1f}%")
    print(f"  Skip above-ATH:         EV = {skip_ev:+.1f}% (delta: {skip_ev - baseline_ev:+.1f}%)")
    print(f"  Stop at ATH:            EV = {stop_ev:+.1f}% (delta: {stop_ev - baseline_ev:+.1f}%)")
    print(f"  Skip + Stop:            EV = {combo_ev:+.1f}% (delta: {combo_ev - baseline_ev:+.1f}%)")

    # By cap breakdown
    print(f"\n--- By Cap: Scenario 4 (Skip + Stop) ---")
    for cap in ['Small', 'Medium', 'Large']:
        cap_df = combo[combo['cap'] == cap]
        if len(cap_df) == 0:
            continue
        n = len(cap_df)
        wr = (cap_df['ath_stop_return'] < 0).mean() * 100
        ev = -cap_df['ath_stop_return'].mean() * 100
        blowup = (cap_df['ath_stop_return'] > 0.10).mean() * 100

        # Compare to baseline
        cap_base = gf_valid[gf_valid['cap'] == cap]
        base_ev = -cap_base['fade_day_return'].mean() * 100
        print(f"  {cap}: n={n}, WR={wr:.0f}%, EV={ev:+.1f}% (was {base_ev:+.1f}%), Blowups={blowup:.0f}%")

    # Yearly breakdown for Scenario 4
    print(f"\n--- Yearly: Scenario 4 (Skip + Stop) vs Baseline ---")
    combo['year'] = combo['date'].str[:4]
    gf_valid['year'] = gf_valid['date'].str[:4]
    for year in sorted(combo['year'].unique()):
        yr_combo = combo[combo['year'] == year]
        yr_base = gf_valid[gf_valid['year'] == year]
        if len(yr_combo) == 0:
            continue
        n = len(yr_combo)
        wr = (yr_combo['ath_stop_return'] < 0).mean() * 100
        ev = -yr_combo['ath_stop_return'].mean() * 100
        base_wr = (yr_base['fade_day_return'] < 0).mean() * 100
        base_ev = -yr_base['fade_day_return'].mean() * 100
        print(f"  {year}: n={n:>3}, WR={wr:.0f}% (was {base_wr:.0f}%), EV={ev:+.1f}% (was {base_ev:+.1f}%)")


if __name__ == '__main__':
    main()
