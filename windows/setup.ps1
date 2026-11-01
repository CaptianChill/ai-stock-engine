# AI Stock Engine - one-time Windows setup.
# Run in PowerShell:  powershell -ExecutionPolicy Bypass -File setup.ps1
# Safe to run again: it updates the code and re-registers the tasks, and only asks for keys
# if they are missing.

$ErrorActionPreference = "Stop"
$Root = Join-Path $env:USERPROFILE "ai-stock-engine"
$Repo = "https://github.com/CaptianChill/ai-stock-engine.git"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}

# 1. Python and Git -----------------------------------------------------------------
Step "Checking Python 3.12 and Git"
$havePy = $false
try { & py -3.12 --version 2>$null | Out-Null; if ($LASTEXITCODE -eq 0) { $havePy = $true } } catch {}
if (-not $havePy) {
    Write-Host "Installing Python 3.12 (winget)..."
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --silent
    Refresh-Path
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Git (winget)..."
    winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements --silent
    Refresh-Path
}

# 2. Code ---------------------------------------------------------------------------
Step "Getting the code into $Root"
if (Test-Path (Join-Path $Root ".git")) {
    git -C $Root pull --ff-only
} else {
    git clone $Repo $Root
}
Set-Location $Root
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

# 3. Python environment -------------------------------------------------------------
Step "Installing Python packages"
if (-not (Test-Path ".venv")) { & py -3.12 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# 4. Keys (you type them; they stay in .env on this PC only) -------------------------
$envFile = Join-Path $Root ".env"
function Ask($name, $hint) {
    $sec = Read-Host "$name ($hint)" -AsSecureString
    $b = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
    try { return ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($b)).Trim() }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b) }
}
if (-not (Test-Path $envFile)) {
    Step "Enter your keys (typing is hidden; paste with right-click, then Enter)"
    $db  = Ask "DATABASE_URL"      "Neon connection string, starts with postgresql://"
    $ak  = Ask "ALPACA_API_KEY"    "Alpaca paper Key"
    $as  = Ask "ALPACA_API_SECRET" "Alpaca paper Secret"
    $an  = Ask "ANTHROPIC_API_KEY" "starts with sk-ant-"
    $lines = @(
        "DATABASE_URL=$db", "ALPACA_API_KEY=$ak", "ALPACA_API_SECRET=$as", "ANTHROPIC_API_KEY=$an",
        "LLM_MODEL=claude-haiku-5-5", "LLM_PROMPT_VERSION=p1", "LLM_KNOWLEDGE_CUTOFF=2026-07-01",
        "BRAIN_MODEL=claude-sonnet-5-5", "ALPACA_DATA_FEED=iex", "TRADING_MODE=paper"
    )
    # UTF-8 without BOM so the keys are read exactly as typed.
    [IO.File]::WriteAllText($envFile, ($lines -join "`r`n") + "`r`n", (New-Object Text.UTF8Encoding($false)))
    # Only your Windows user can read the key file.
    icacls $envFile /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null
} else {
    Write-Host "Keys already saved in $envFile (delete that file and re-run to change them)."
}

# 5. Checks -------------------------------------------------------------------------
Step "Running the test suite"
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "tests failed" }
Step "Checking the database connection"
& .\.venv\Scripts\python.exe -m engine init-db
if ($LASTEXITCODE -ne 0) { throw "database connection failed: check DATABASE_URL in $envFile" }
& .\.venv\Scripts\python.exe -m engine status

# 6. Scheduled tasks ----------------------------------------------------------------
Step "Registering Windows scheduled tasks"
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -MultipleInstances IgnoreNew `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$days = "Monday","Tuesday","Wednesday","Thursday","Friday"

# Trader: every 10 minutes on weekdays, 6:30 AM - 5:30 PM local time. The engine itself reads
# the New York clock and runs only the stage that is due, once per day.
$tick = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At 6:30am
$tick.Repetition = (New-ScheduledTaskTrigger -Once -At 6:30am `
    -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Hours 11)).Repetition
$tickAction = New-ScheduledTaskAction -Execute (Join-Path $Root "windows\tick.cmd") -WorkingDirectory $Root
Register-ScheduledTask -TaskName "AI Stock Engine - Trader" -Trigger $tick -Action $tickAction `
    -Settings $settings -Description "Paper trading tick every 10 minutes on weekdays" -Force | Out-Null

# Review brain: weekdays 6:15 PM local (after the 4:30 PM New York close stage).
$brain = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At 6:15pm
$brainAction = New-ScheduledTaskAction -Execute (Join-Path $Root "windows\brain.cmd") -WorkingDirectory $Root
Register-ScheduledTask -TaskName "AI Stock Engine - Brain" -Trigger $brain -Action $brainAction `
    -Settings $settings -Description "Nightly review brain" -Force | Out-Null

Get-ScheduledTask -TaskName "AI Stock Engine*" | Select-Object TaskName, State | Format-Table

Step "Done"
Write-Host "Next: start the first validation run (takes a few hours):"
Write-Host "    $Root\windows\research.cmd" -ForegroundColor Yellow
Write-Host "Logs go to $Root\logs. Keep this PC on (or asleep with wake timers allowed) on weekdays."
