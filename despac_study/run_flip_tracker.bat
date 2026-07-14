@echo off
cd /d C:\Users\zmbur\PycharmProjects\backtester
venv\Scripts\python.exe despac_study\flip_tracker.py >> despac_study\cron_win.log 2>&1
