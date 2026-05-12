@echo off
cd /d "C:\Users\zmbur\PycharmProjects\backtester" || goto :fail

REM use the SAME venv as generate_report
call "C:\Users\zmbur\PycharmProjects\contextAI\venv\Scripts\activate.bat" || goto :fail

set "PYTHONPATH=%CD%"

REM Open browser after a short delay, then start Streamlit
start "" cmd /c "timeout /t 3 >nul & start http://localhost:8501"
python -m streamlit run dashboard\app.py --server.headless true || goto :fail
goto :eof

:fail
echo [%date% %time%] ERROR %errorlevel%
exit /b %errorlevel%
