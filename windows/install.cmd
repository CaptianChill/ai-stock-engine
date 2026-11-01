@echo off
rem Double-click this file to set up the AI Stock Engine on this PC.
echo Downloading and running the AI Stock Engine setup...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/CaptianChill/ai-stock-engine/main/windows/setup.ps1 -OutFile $env:TEMP\ase-setup.ps1; & $env:TEMP\ase-setup.ps1"
echo.
pause
