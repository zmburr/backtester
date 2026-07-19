@echo off
cd /d "C:\Users\zmbur\PycharmProjects\backtester" || goto :fail
call "C:\Users\zmbur\PycharmProjects\backtester\venv\Scripts\activate.bat" || goto :fail
set "PYTHONPATH=%CD%"

REM --- Fill forward outcomes on the signal ledger for prior settled days (idempotent) ---
python -m scripts.fill_signal_outcomes >> "%~dp0priority_report.log" 2>&1

python scripts\priority_report.py >> "%~dp0priority_report.log" 2>&1 || goto :fail

REM --- TTS alert on success (line written by priority_report.py carries the
REM --- top intensity score + cluster flag; falls back to the generic line) ---
powershell -Command "Add-Type -AssemblyName System.Speech; $t = Get-Content -Raw 'data\priority_signals\latest_tts.txt' -ErrorAction SilentlyContinue; if (-not $t) { $t = 'Priority report sent to email' }; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($t)"
goto :eof

:fail
echo [%date% %time%] ERROR %errorlevel% >> "%~dp0priority_report.log"
powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('Priority report failed')"
exit /b %errorlevel%
