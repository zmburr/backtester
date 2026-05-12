@echo off
cd /d "C:\Users\zmbur\PycharmProjects\backtester" || goto :fail

REM use contextAI venv (has xbbg + blpapi for Bloomberg EQS screens)
call "C:\Users\zmbur\PycharmProjects\contextAI\venv\Scripts\activate.bat" || goto :fail

set "PYTHONPATH=%CD%"
python scripts\screener_report.py >> "%~dp0screener_report.log" 2>&1 || goto :fail
goto :eof
:fail
echo [%date% %time%] ERROR %errorlevel% >> "%~dp0screener_report.log"
exit /b %errorlevel%
