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
rem One time only, on PCs set up before hidden.vbs existed: stop the tasks opening a black window.
if not exist logs\hidden-tasks.done (
  powershell -NoProfile -ExecutionPolicy Bypass -File windows\hide-tasks.ps1 >> logs\tick.log 2>&1
  echo done> logs\hidden-tasks.done
)
.venv\Scripts\python.exe -m engine tick >> logs\tick.log 2>&1
echo ----- finished %time% (exit %errorlevel%) >> logs\tick.log
