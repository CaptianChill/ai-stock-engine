@echo off
rem Called by the "AI Stock Engine - Brain" scheduled task every weekday evening.
cd /d "%~dp0.."
echo ===== %date% %time% >> logs\brain.log
.venv\Scripts\python.exe -m engine brain >> logs\brain.log 2>&1
.venv\Scripts\python.exe -m engine report >> logs\brain.log 2>&1
.venv\Scripts\python.exe -m engine proposals >> logs\brain.log 2>&1
