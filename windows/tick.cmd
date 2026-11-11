@echo off
rem Called by the "AI Stock Engine - Trader" scheduled task every 10 minutes on weekdays.
cd /d "%~dp0.."
echo ===== %date% %time% >> logs\tick.log
rem Pick up fixes pushed to GitHub without anyone having to touch this PC (fast-forward only).
git pull --ff-only -q >> logs\tick.log 2>&1
.venv\Scripts\python.exe -m engine tick >> logs\tick.log 2>&1
