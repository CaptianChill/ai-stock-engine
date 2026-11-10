@echo off
rem Called by the "AI Stock Engine - Brain" scheduled task every weekday evening.
cd /d "%~dp0.."
echo ===== %date% %time% >> logs\brain.log
rem Nightly self-update: keep this PC an exact copy of GitHub (.env and logs are untouched).
git fetch origin main >> logs\brain.log 2>&1 && git reset --hard origin/main >> logs\brain.log 2>&1
.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet >> logs\brain.log 2>&1
.venv\Scripts\python.exe -m engine brain >> logs\brain.log 2>&1
.venv\Scripts\python.exe -m engine report --notify >> logs\brain.log 2>&1
.venv\Scripts\python.exe -m engine proposals >> logs\brain.log 2>&1
