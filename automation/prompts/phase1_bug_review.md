# Phase 1: Bug Hunt & Fix

You are running as a scheduled daily code review for the Backtester project. Today's date will be provided by the calling script.

## Instructions

### 1. Create a Feature Branch

```
git checkout -b automation/bug-review-{TODAY_DATE}
```

Where `{TODAY_DATE}` is today's date in YYYY-MM-DD format.

### 2. Systematic Code Review

Read through the codebase in this priority order. For each file, read it completely before moving to the next.

**Priority 1 — Trading Logic (highest impact)**
- `backtesting_strategies/entry_signals.py`
- `backtesting_strategies/exit_signals.py`
- `backtesting_strategies/stop_strategies.py`
- `backtesting_strategies/trade.py`

**Priority 2 — Scoring & Analysis**
- `analyzers/bounce_scorer.py`
- `analyzers/bounce_exit_targets.py`
- `analyzers/reversal_scorer.py`
- `analyzers/reversal_pretrade.py`
- `analyzers/charter.py`

**Priority 3 — Data Pipeline**
- `data_collectors/combined_data_collection.py`
- `data_collectors/bounce_collector.py`
- `data_collectors/strong_bounce_collector.py`
- `data_queries/polygon_queries.py`
- `data_queries/trillium_queries.py`

**Priority 4 — Scanners**
- `scanners/setup_screener.py`
- `scanners/historical_backscanner.py`
- `scanners/stock_screener.py`
- `scanners/live_watcher.py`
- `scanners/bounce_trader.py`

**Priority 5 — Reports & Support**
- `scripts/generate_report.py`
- `support/config.py`
- `support/llm_client.py`

**Priority 6 — Backtest Runner**
- `backtesters/main_backtester.py`

### 3. What to Look For

For each file, specifically check for:

- **Logic bugs**: off-by-one errors, wrong comparisons, inverted conditions, incorrect operator precedence
- **Null/NaN handling**: missing checks on DataFrame columns, unsafe `.iloc`/`.loc` access, NaN propagation in calculations
- **Division by zero**: unguarded divisions, especially in percentage calculations and ATR-based metrics
- **API error handling gaps**: Polygon API calls without try/except, missing rate limit handling, no retry logic for transient failures
- **Pandas pitfalls**: chained assignment warnings, SettingWithCopyWarning patterns, incorrect `inplace` usage, comparing with `==` instead of `.equals()` or `pd.isna()`
- **Dead code**: unreachable branches, unused imports, commented-out blocks that should be removed
- **Bare excepts**: `except:` or `except Exception:` that silently swallow errors
- **Type mismatches**: string/int comparisons, datetime vs string dates, float vs int edge cases
- **Race conditions**: file I/O without proper guards, shared state issues in threaded code (especially `bounce_trader.py`)

### 4. What to Fix

**DO fix:**
- Clear, unambiguous bugs
- Missing null/NaN guards that could cause runtime errors
- Bare excepts that hide real errors (replace with specific exception types + logging)
- Division-by-zero vulnerabilities
- Dead code that serves no purpose

**DO NOT fix (flag in report only):**
- Style issues or code formatting
- Refactoring opportunities (that's Phase 2)
- Threshold value changes (trading parameters are intentional)
- CSV column changes or config file edits
- Anything where the "fix" requires domain judgment about trading strategy

### 5. Commit and Push

After making all fixes:

```
git add -A
git commit -m "automation: daily bug review {TODAY_DATE} — {N} bugs fixed"
git push -u origin automation/bug-review-{TODAY_DATE}
```

Then attempt to create a PR:
```
gh pr create --title "Daily Bug Review {TODAY_DATE}" --body "Automated bug review. See commit details for fixes."
```

If `gh` fails (not installed, auth issues), that's fine — skip it silently.

### 6. Output Report

**CRITICAL**: After completing all code changes, commits, and pushes, you MUST output your complete findings as plain text in your final response. This text is captured to a log file. If you do not output the report, the log file will be empty. Do NOT use the Write tool for this — just output the text directly.

Output your complete findings as a structured markdown report:

```markdown
# Daily Bug Review — {TODAY_DATE}

## Bugs Fixed

| # | File | Line(s) | Bug Description | Fix Applied |
|---|------|---------|-----------------|-------------|
| 1 | ... | ... | ... | ... |

## Uncertain Issues (Flagged Only)

Issues that might be bugs but require human judgment:

| # | File | Line(s) | Description | Why Uncertain |
|---|------|---------|-------------|---------------|
| 1 | ... | ... | ... | ... |

## Clean Files

Files reviewed with no issues found:
- file1.py
- file2.py
- ...

## Summary

- Files reviewed: X
- Bugs fixed: Y
- Issues flagged: Z
- Branch: automation/bug-review-{TODAY_DATE}
- PR created: Yes/No
```

If no bugs were found, still output the report with "Bugs Fixed" showing 0 and list all reviewed files under "Clean Files".
