@echo off
cd /d "C:\Users\zmbur\PycharmProjects\backtester" || goto :fail

REM use the SAME venv PyCharm shows
call "C:\Users\zmbur\PycharmProjects\contextAI\venv\Scripts\activate.bat" || goto :fail

set "PYTHONPATH=%CD%"
python scripts\generate_report.py >> "%~dp0generate_report.log" 2>&1 || goto :fail
goto :eof
:fail
echo [%date% %time%] ERROR %errorlevel% >> "%~dp0generate_report.log"
exit /b %errorlevel%
