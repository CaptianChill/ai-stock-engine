@echo off
rem Quick look: audit gates, kill switch, positions, latest brain report, last trader log lines.
cd /d "%~dp0.."
.venv\Scripts\python.exe -m engine status
echo.
.venv\Scripts\python.exe -m engine report
echo.
echo ---- last trader log lines ----
powershell -NoProfile -Command "if (Test-Path logs\tick.log) { Get-Content logs\tick.log -Tail 15 }"
pause
