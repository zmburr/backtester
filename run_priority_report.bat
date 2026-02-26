@echo off
cd /d "C:\Users\zmbur\PycharmProjects\backtester" || goto :fail
call "C:\Users\zmbur\PycharmProjects\contextAI\venv\Scripts\activate.bat" || goto :fail
set "PYTHONPATH=%CD%"
python scripts\priority_report.py >> "%~dp0priority_report.log" 2>&1 || goto :fail
goto :eof
:fail
echo [%date% %time%] ERROR %errorlevel% >> "%~dp0priority_report.log"
exit /b %errorlevel%
