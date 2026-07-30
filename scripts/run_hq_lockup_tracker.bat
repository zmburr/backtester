@echo off
REM run_hq_lockup_tracker.bat — daily HQ warrant-redemption / lockup progress check.
REM Triggered by Task Scheduler at 4:15pm ET on weekdays. Also double-clickable.

setlocal

title === HQ LOCKUP TRACKER (backtester) ===

set BT_ROOT=C:\Users\zmbur\PycharmProjects\backtester
set LOG_DIR=%BT_ROOT%\scripts\logs

REM Locale-independent YYYY-MM-DD via wmic
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set DT=%%I
set DATE_TAG=%DT:~0,4%-%DT:~4,2%-%DT:~6,2%

set LOG_FILE=%LOG_DIR%\hq_lockup_%DATE_TAG%.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

cd /d "%BT_ROOT%"

echo [%DATE% %TIME%] launching HQ lockup tracker >> "%LOG_FILE%"
"%BT_ROOT%\venv\Scripts\python.exe" scripts\hq_lockup_tracker.py >> "%LOG_FILE%" 2>&1
set EXITCODE=%ERRORLEVEL%

echo [%DATE% %TIME%] HQ lockup tracker exited (code %EXITCODE%) >> "%LOG_FILE%"

REM Propagate the exit code so Task Scheduler's LastResult reflects failures.
endlocal & exit /b %EXITCODE%
