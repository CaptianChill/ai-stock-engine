@echo off
rem Backtest: trades v1.4.0 (current) and v1.5.0-ml (scikit-learn) over every day since March 2017
rem as if live, costs included, audits both, and compares them with simply holding SPY.
rem The scoreboard is also sent to your phone. Reuse prices already downloaded:  backtest.cmd --skip-fetch
cd /d "%~dp0.."
if not exist logs mkdir logs
echo Getting the latest version of the engine...
git pull --ff-only
.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
echo Running the backtest (downloading prices can take an hour the first time). Progress: logs\backtest.log
.venv\Scripts\python.exe -u -m engine compare %* >> logs\backtest.log 2>&1
echo.
powershell -NoProfile -Command "Get-Content logs\backtest.log -Tail 12"
echo.
echo Full log: %cd%\logs\backtest.log
pause
