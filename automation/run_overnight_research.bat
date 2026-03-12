@echo off
REM Overnight Backtester Researcher — Task Scheduler wrapper
REM Schedule: 10 PM ET daily via Windows Task Scheduler

cd /d "C:\Users\zmbur\PycharmProjects\backtester"

REM Activate venv and run
call venv\Scripts\activate.bat
python scripts\run_overnight_research.py --max-iterations 15 --max-runtime 7200

REM Deactivate
deactivate
