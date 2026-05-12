# register_priority_report.ps1 -- Register the morning priority report in Windows Task Scheduler.
# Fires at 9:00 AM ET on weekdays so the morning_watcher (which fires at 9:25)
# has today's signal JSON ready when it launches.
#
# Run from a PowerShell prompt (admin not required):
#   cd C:\Users\zmbur\PycharmProjects\backtester
#   Set-ExecutionPolicy -Scope Process Bypass -Force
#   .\scripts\register_priority_report.ps1

$taskName = "backtester-PriorityReport-Morning"
$description = "Morning priority report -- scans watchlist, emails report, writes signal JSON for morning_watcher"
$batPath = "C:\Users\zmbur\PycharmProjects\backtester\run_priority_report.bat"
$workingDir = "C:\Users\zmbur\PycharmProjects\backtester"

$action = New-ScheduledTaskAction `
    -Execute $batPath `
    -WorkingDirectory $workingDir

# Weekly Mon-Fri at 9:00 AM (host clock should be ET).
$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday `
    -At 9:00AM

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
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
Write-Host "Task '$taskName' registered."
Write-Host "Schedule: Mon-Fri at 9:00 AM (ET if host clock is ET)"
Write-Host "Time limit: 10 minutes"
Write-Host "Logs: C:\Users\zmbur\PycharmProjects\backtester\priority_report.log"
Write-Host ""
Write-Host "To verify:    Get-ScheduledTask -TaskName '$taskName' | Format-List"
Write-Host "To run now:   Start-ScheduledTask -TaskName '$taskName'"
Write-Host "To remove:    Unregister-ScheduledTask -TaskName '$taskName'"
