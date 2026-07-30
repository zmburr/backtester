@echo off
cd /d "C:\Users\zmbur\PycharmProjects\backtester" || goto :fail
call "C:\Users\zmbur\PycharmProjects\backtester\venv\Scripts\activate.bat" || goto :fail
set "PYTHONPATH=%CD%"

REM --- Evening bounce board -> signal ledger (rows dated next trading day;
REM --- outcomes filled by the morning bat's fill_signal_outcomes pass) ---
python -m scripts.evening_signal_log >> "%~dp0evening_signal_log.log" 2>&1 || goto :fail
goto :eof

:fail
echo [%date% %time%] ERROR %errorlevel% >> "%~dp0evening_signal_log.log"
exit /b %errorlevel%
