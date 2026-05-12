@echo off
REM Runs EQS Screener Report 10 minutes after logon (weekdays only).
REM Place a shortcut to this file in shell:startup.

REM Check if today is a weekday (Mon=1 .. Fri=5, Sat=6, Sun=0)
for /f %%d in ('powershell -NoProfile -Command "(Get-Date).DayOfWeek.value__"') do set DOW=%%d
if %DOW%==0 exit /b 0
if %DOW%==6 exit /b 0

REM Wait 10 minutes (600 seconds)
timeout /t 600 /nobreak >nul

REM Run the report
call "C:\Users\zmbur\PycharmProjects\backtester\run_screener_report.bat"
