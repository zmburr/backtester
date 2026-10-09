# register_dead_lows_scanner.ps1 -- Register the dead-lows close scanner in Windows Task Scheduler.
# Fires at 12:30 PM ET on weekdays; the scanner builds its daily context at launch,
# then sleeps until close-13 / close-8 min (NYSE calendar, so half-days work) to
# check, alert and log. Needs this desktop (Trillium/SHEL) and the user logged on
# for the spoken alert.
#
# Run from a PowerShell prompt (admin not required):
#   cd C:\Users\zmbur\PycharmProjects\backtester
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   .\scripts\register_dead_lows_scanner.ps1

$taskName = "backtester-DeadLowsScanner"
$description = "Dead-lows close scanner -- overnight starter alert (speaks >=1B tier, emails the rest, logs to signal ledger)"
$batPath = "C:\Users\zmbur\PycharmProjects\backtester\run_dead_lows_scanner.bat"
$workingDir = "C:\Users\zmbur\PycharmProjects\backtester"

$action = New-ScheduledTaskAction `
    -Execute $batPath `
    -WorkingDirectory $workingDir

# Weekly Mon-Fri at 12:30 PM (host clock should be ET).
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At 12:30PM

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed existing task: $taskName"
}

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description $description `
    -RunLevel Limited

Write-Host ""
Write-Host "Registered: $taskName (weekdays 12:30 PM ET -> $batPath)"
