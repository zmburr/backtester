"""IV top-timing study: does IV acceleration predict when a parabolic stock tops out?

Pipeline (see PLAN.md):
    fetch_iv.py       -- phase 01: per-trade intraday ATM IV series -> data/iv_study/
    build_features.py -- phase 02: parquets -> iv_features.csv        (pending)
    event_study.py    -- phase 03: alignment, key statistic, report   (pending)
    pilot_plot.py     -- throwaway overlay plots for the Phase 1 pilot gate

Run from the project root, e.g.:
    venv/Scripts/python.exe -m iv_study.fetch_iv --pilot
"""
