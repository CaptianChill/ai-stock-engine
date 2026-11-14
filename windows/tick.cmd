@echo off
rem Called by the "AI Stock Engine - Trader" scheduled task every 10 minutes on weekdays.
cd /d "%~dp0.."
rem UTF-8 output; never let git stop to ask for a password; give up on a stalled download.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set GIT_TERMINAL_PROMPT=0
set GIT_HTTP_LOW_SPEED_LIMIT=1000
set GIT_HTTP_LOW_SPEED_TIME=30
echo ===== %date% %time% >> logs\tick.log
rem Pick up fixes pushed to GitHub without anyone having to touch this PC (fast-forward only).
git pull --ff-only -q >> logs\tick.log 2>&1
.venv\Scripts\python.exe -m engine tick >> logs\tick.log 2>&1
echo ----- finished %time% (exit %errorlevel%) >> logs\tick.log
