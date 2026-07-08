@echo off
REM Premarket Bounce Scanner - schedule at ~4:10 AM ET on trading days.
REM Loops until 9:30 ET then exits. Alerts are EMAIL ONLY (remote-machine safe).
cd /d "C:\Users\zmbur\PycharmProjects\backtester" || goto :fail
call "C:\Users\zmbur\PycharmProjects\backtester\venv\Scripts\activate.bat" || goto :fail
set "PYTHONPATH=%CD%"
python -m scanners.premarket_bounce_scanner >> "%~dp0premarket_scanner.log" 2>&1 || goto :fail
goto :eof

:fail
echo [%date% %time%] ERROR %errorlevel% >> "%~dp0premarket_scanner.log"
exit /b %errorlevel%
