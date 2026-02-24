@echo off
REM Daily code review - runs at 5 PM MT (7 PM ET) via Task Scheduler
REM Phase 1: Bug hunt + fix on feature branch
REM Phase 2: Read-only recommendations report

cd /d C:\Users\zmbur\PycharmProjects\backtester

REM Clear nested-session guard so this works even when tested from inside Claude Code
set CLAUDECODE=

REM Create dirs if missing
if not exist automation\logs mkdir automation\logs
set VAULT_DIR=C:\Users\zmbur\OneDrive\Documents\Obsidian Vault\Code Review
if not exist "%VAULT_DIR%" mkdir "%VAULT_DIR%"

REM Timestamp for log files
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set LOGDATE=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%

echo [%date% %time%] Starting daily code review... >> automation\logs\daily_review.log

REM ============================================================
REM  GIT SETUP: stash work, update master, tag for rollback
REM ============================================================

REM Stash any uncommitted work
git stash push -m "daily-review-autostash-%LOGDATE%"

REM Switch to master and pull latest
git checkout master
git pull

REM Tag current state for rollback safety
git tag pre-review-%LOGDATE% 2>nul

REM ============================================================
REM  PHASE 1: Bug Hunt (on feature branch)
REM ============================================================

echo [%date% %time%] Starting Phase 1: Bug Review... >> automation\logs\daily_review.log

C:\Users\zmbur\.local\bin\claude.exe --print --dangerously-skip-permissions --output-format text ^
  --max-turns 80 ^
  -p "You are running as a scheduled daily code review for the Backtester project. Today is %LOGDATE%. Read the file automation/prompts/phase1_bug_review.md and follow its instructions exactly. Replace all occurrences of {TODAY_DATE} with %LOGDATE%. IMPORTANT: After all code changes and git operations are done, you MUST output your full structured bug review report as plain text in your final response. This text gets captured to a log file. Do not skip the report." ^
  > "%VAULT_DIR%\Backtester bug review %LOGDATE%.md" 2>&1

echo [%date% %time%] Phase 1 complete. Output: %VAULT_DIR%\Backtester bug review %LOGDATE%.md >> automation\logs\daily_review.log

REM ============================================================
REM  Return to master for Phase 2
REM ============================================================

git checkout master

REM ============================================================
REM  PHASE 2: Recommendations (read-only, stays on master)
REM ============================================================

echo [%date% %time%] Starting Phase 2: Recommendations... >> automation\logs\daily_review.log

C:\Users\zmbur\.local\bin\claude.exe --print --dangerously-skip-permissions --output-format text ^
  --max-turns 50 ^
  -p "You are running as a scheduled daily code review for the Backtester project. Today is %LOGDATE%. Read the file automation/prompts/phase2_recommendations.md and follow its instructions exactly. Replace all occurrences of {TODAY_DATE} with %LOGDATE%. Do NOT modify any files - this is a read-only review. Output your full report as your final response." ^
  > "%VAULT_DIR%\Backtester code review %LOGDATE%.md" 2>&1

echo [%date% %time%] Phase 2 complete. Output: %VAULT_DIR%\Backtester code review %LOGDATE%.md >> automation\logs\daily_review.log

REM ============================================================
REM  CLEANUP: restore user's work
REM ============================================================

REM Pop the stash if one was created (ignore error if no stash exists)
git stash pop 2>nul

echo [%date% %time%] Daily review completed. >> automation\logs\daily_review.log
