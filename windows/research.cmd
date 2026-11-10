@echo off
rem First validation run: ~10 years of prices, news since the LLM cutoff, LLM scoring (all headlines,
rem half price through Anthropic's batch service), replay with costs, every audit gate, kill-switch self-test. Takes a few hours.
rem To reuse data already downloaded:  research.cmd --skip-fetch
cd /d "%~dp0.."
if not exist logs mkdir logs
echo Getting the latest version of the engine...
git pull --ff-only
.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
echo Running the validation run. This takes a few hours; progress is in logs\research.log
echo DO NOT CLOSE THIS WINDOW. If it gets closed, just run research.cmd again: it resumes.
.venv\Scripts\python.exe -u -m engine research 2026-07-02 --batch %* >> logs\research.log 2>&1
echo.
echo ---- last 40 lines ----
powershell -NoProfile -Command "Get-Content logs\research.log -Tail 40"
echo.
echo Finished. Full log: %cd%\logs\research.log
pause
