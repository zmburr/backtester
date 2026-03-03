@echo off
cd /d "C:\Users\zmbur\PycharmProjects\backtester" || goto :fail
call "C:\Users\zmbur\PycharmProjects\backtester\venv\Scripts\activate.bat" || goto :fail
set "PYTHONPATH=%CD%"
python scripts\priority_report.py >> "%~dp0priority_report.log" 2>&1 || goto :fail

REM --- TTS alert on success ---
powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('Priority report sent to email')"
goto :eof

:fail
echo [%date% %time%] ERROR %errorlevel% >> "%~dp0priority_report.log"
powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('Priority report failed')"
exit /b %errorlevel%
