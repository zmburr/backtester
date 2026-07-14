# De-SPAC Event Study (2020-present)

## Research questions

1. Which SPAC deals completed (de-SPACed) since 2020, and which targets are
   actually interesting (high-tech) businesses?
2. What happens to the stock at each deal milestone: **announcement**,
   **shareholder vote / redemptions**, **merger close / ticker flip**?
3. How much do redemptions matter (tiny post-redemption float -> squeeze)?
4. Is there a repeatable trade buying the last close under the OLD ticker,
   before the symbol flips overnight?

## Data sources

- **SEC EDGAR** full-text search + submissions JSON + filing texts (free,
  10 req/s, User-Agent required). SPACs file under SIC 6770; completion =
  8-K Item 2.01 ("super 8-K"); announcement = first Form 425; vote = Item
  5.07 8-K; redemption counts parsed from filing text.
- **Polygon**: `/v3/reference/tickers?cik=&date=` resolves the exact symbol a
  registrant traded under on any date (authoritative old-ticker / flip-ticker
  mapping); `/vX/reference/tickers/{T}/events` for explicit flip dates;
  first UNADJUSTED daily bar under the new symbol as fallback flip date.
  **Bars must be UNadjusted**: later reverse splits are baked into the new
  symbol's adjusted history but not the old symbol's, which fabricates
  30-50x flip returns.

## Pipeline (run in order; everything disk-cached and resumable)

```
python -m despac_study.build_universe    # EDGAR FTS sweep -> despac_universe.csv
python -m despac_study.edgar_timeline    # dates + redemptions -> despac_timeline.csv
python -m despac_study.polygon_enrich    # tickers/flip/floats -> despac_enriched.csv
python -m despac_study.classify          # sector + is_hightech -> despac_classified.csv
python -m despac_study.fetch_prices      # event metrics -> despac_data.csv  (MASTER)
python -m despac_study.analysis          # tables + charts -> reports/despac_study/despac_report.html
```

Caches: `data/despac_study/edgar_cache/` (FTS pages, submissions, filing
texts) and `data/despac_study/cache/` (Polygon JSON pickles). Delete to force
refetch; otherwise reruns are free.

## Key columns in despac_data.csv

- identity: cik, company_name, old_ticker, flip_ticker, new_ticker, sector,
  is_hightech
- timeline: ann_date, vote_date, close_date, flip_date, flip_lag_cal_days
- redemptions: redeemed_shares, redemption_pct_best (stated % preferred,
  else redeemed / pre-vote Class A shares from Polygon), post_redeem_float_est
- announcement: ann_gap_pct, ann_day0_ret_pct, ann_2d_high_ret_pct,
  ann_to_vote_ret_pct
- flip trade (entry = last_old_close): flip_gap_pct, flip_day_ret_pct,
  flip_high_ret_pct, post_flip_ret_{1,3,5,10}d_pct, max_runup_10d_pct,
  max_drawdown_10d_pct

## Live flip tracker

`flip_tracker.py` — standalone (stdlib + requests), cron-friendly single shot.
Sweeps EDGAR for last-12-day 8-Ks with trust language + Item 5.07/2.01,
extracts redemptions, tags hot themes, pulls price-vs-trust from Polygon,
scores the checklist, emails NEW candidates (state file dedupe).

- Windows (stopgap): Task Scheduler `DespacFlipTracker`, every 30 min
  07:00-20:30 via `run_flip_tracker.bat` -> `cron_win.log`.
  Remove with `schtasks /Delete /TN DespacFlipTracker /F` once Mac is live.
- Mac (primary): bundle synced to `~/ObsidianVault/despac_tracker/`;
  `bash install.sh` installs cron (*/30 5-18 MT weekdays) at ~/despac_tracker.
- Keys: `.env` beside script (POLYGON_API_KEY, GMAIL_PASSWORD).

## Known limitations

- ~10% of deals list the target under a brand-new CIK (double-dummy, e.g.
  Payoneer): no pre-close filings under that CIK -> old_ticker missing ->
  announcement/flip-entry metrics unavailable for those.
- Redemption % coverage is partial (regex over filing text); deals with no
  stated count and no Polygon pre-vote share count stay NaN.
- Non-SPAC 8-K matches (ordinary M&A) are filtered by is_spac (425s / SIC
  6770 / rename near close) plus the no-flip-event exclusion in analysis.
