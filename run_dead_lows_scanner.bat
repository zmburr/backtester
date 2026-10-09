@echo off
cd /d "C:\Users\zmbur\PycharmProjects\backtester" || goto :fail
call "C:\Users\zmbur\PycharmProjects\backtester\venv\Scripts\activate.bat" || goto :fail
set "PYTHONPATH=%CD%"

REM --- Dead-lows close scanner: builds context at launch, checks at close-13
REM --- and close-8, speaks/emails new names, logs candidates to the ledger ---
python -m scanners.dead_lows_scanner >> "%~dp0dead_lows_scanner.log" 2>&1 || goto :fail
goto :eof

:fail
echo [%date% %time%] ERROR %errorlevel% >> "%~dp0dead_lows_scanner.log"
exit /b %errorlevel%
