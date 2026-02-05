## Backtester Checklist Dashboard

Interactive Streamlit dashboard for exploring:
- `data/bounce_data.csv` (long bounces off capitulation sell-offs)
- `data/reversal_data.csv` (short parabolic reversals)

It also **simulates your checklist scoring** so you can tune thresholds and immediately see how GO/CAUTION/NO-GO stats change on historical trades.

### Run

From the repo root:

```bash
streamlit run dashboard/app.py
```

Or on Windows, you can run `run_dashboard.bat` (it will prefer `venv\Scripts\python.exe` if present).

### What it mirrors

- **Bounce checklist**: uses `analyzers/bounce_scorer.py` `BouncePretrade` + `SETUP_PROFILES`.
- **Reversal pre-trade checklist (5 criteria)**: mirrors the `score_pretrade_setup()` logic in `scripts/generate_report.py`.
  - The thresholds are currently duplicated in `dashboard/app.py` as `REVERSAL_PRETRADE_THRESHOLDS` so the dashboard can run without importing the full report script.

