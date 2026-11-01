@echo off
rem Called by the "AI Stock Engine - Trader" scheduled task every 10 minutes on weekdays.
cd /d "%~dp0.."
echo ===== %date% %time% >> logs\tick.log
.venv\Scripts\python.exe -m engine tick >> logs\tick.log 2>&1
