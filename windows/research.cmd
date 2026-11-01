@echo off
rem First validation run: ~10 years of prices, news since the LLM cutoff, LLM scoring (capped at
rem 5,000 calls), replay with costs, every audit gate, kill-switch self-test. Takes a few hours.
rem To reuse data already downloaded:  research.cmd --skip-fetch
cd /d "%~dp0.."
if not exist logs mkdir logs
echo Running the validation run. This takes a few hours; progress is in logs\research.log
.venv\Scripts\python.exe -u -m engine research 2026-07-02 --max-calls 5000 %* > logs\research.log 2>&1
echo.
echo ---- last 40 lines ----
powershell -NoProfile -Command "Get-Content logs\research.log -Tail 40"
echo.
echo Finished. Full log: %cd%\logs\research.log
pause
