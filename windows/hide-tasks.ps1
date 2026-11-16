# Switches the two scheduled tasks to start through hidden.vbs, so they stop opening a black
# terminal window every 10 minutes. Only the tasks' action changes; schedules and settings stay.
# tick.cmd runs this once by itself on PCs set up before this change; safe to run again by hand:
#   powershell -ExecutionPolicy Bypass -File windows\hide-tasks.ps1
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$vbs = Join-Path $Root "windows\hidden.vbs"
foreach ($t in @(@{Name = "AI Stock Engine - Trader"; Cmd = "tick.cmd"},
                 @{Name = "AI Stock Engine - Brain";  Cmd = "brain.cmd"})) {
    $cmd = Join-Path $Root ("windows\" + $t.Cmd)
    $action = New-ScheduledTaskAction -Execute "wscript.exe" `
        -Argument ("//B //Nologo `"{0}`" `"{1}`"" -f $vbs, $cmd) -WorkingDirectory $Root
    try {
        Set-ScheduledTask -TaskName $t.Name -Action $action | Out-Null
        Write-Output "hidden: $($t.Name)"
    } catch {
        Write-Output "could not update $($t.Name): $($_.Exception.Message)"
    }
}
