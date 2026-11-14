@echo off
rem Called by the "AI Stock Engine - Brain" scheduled task every weekday evening.
cd /d "%~dp0.."
rem UTF-8 output; never let git stop to ask for a password; give up on a stalled download.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set GIT_TERMINAL_PROMPT=0
set GIT_HTTP_LOW_SPEED_LIMIT=1000
set GIT_HTTP_LOW_SPEED_TIME=30
echo ===== %date% %time% >> logs\brain.log
rem Nightly self-update: keep this PC an exact copy of GitHub (.env and logs are untouched).
git fetch origin main >> logs\brain.log 2>&1 && git reset --hard origin/main >> logs\brain.log 2>&1
.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet >> logs\brain.log 2>&1
.venv\Scripts\python.exe -m engine brain >> logs\brain.log 2>&1
.venv\Scripts\python.exe -m engine report --notify >> logs\brain.log 2>&1
.venv\Scripts\python.exe -m engine proposals >> logs\brain.log 2>&1
